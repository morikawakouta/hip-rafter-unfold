let pyodide = null;

const PY_CODE = `
import io, base64, math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

Zhat = np.array([0.0, 0.0, 1.0], dtype=float)

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v*0.0
    return v / n

def make_R_from_axes(x_axis, y_axis, z_axis):
    # columns are axes in world (local->world)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(float)

# ----------------------------
# Prism geometry
# ----------------------------
def build_prism_vertices(w: float, h: float, L: float = 2000.0) -> np.ndarray:
    # local prism: x length, y width, z height (centered)
    y = w/2.0
    z = h/2.0
    x0 = -L/2.0
    x1 =  L/2.0
    verts = []
    for x in (x0, x1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append([x, sy*y, sz*z])
    return np.array(verts, dtype=float)

BOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

def plane_eval(n, d, p) -> float:
    return float(np.dot(n, p) - d)

def intersect_segment_plane(p1, p2, n, d, eps: float = 1e-9):
    f1 = plane_eval(n, d, p1)
    f2 = plane_eval(n, d, p2)
    if abs(f1) < eps and abs(f2) < eps:
        return None
    if abs(f1) < eps:
        return p1.copy()
    if abs(f2) < eps:
        return p2.copy()
    if f1 * f2 > 0:
        return None
    t = f1 / (f1 - f2)
    return p1 + t * (p2 - p1)

def unique_points(points, tol: float = 1e-6) -> np.ndarray:
    uniq = []
    for p in points:
        if p is None:
            continue
        if all(np.linalg.norm(p - q) >= tol for q in uniq):
            uniq.append(p)
    return np.array(uniq, dtype=float)

def plane_basis(n: np.ndarray):
    n = normalize(n)
    a = np.array([0.0,0.0,1.0])
    if abs(np.dot(a,n)) > 0.9:
        a = np.array([0.0,1.0,0.0])
    e1 = normalize(np.cross(n,a))
    e2 = normalize(np.cross(n,e1))
    return e1, e2

def order_polygon(points: np.ndarray, n: np.ndarray) -> np.ndarray:
    if points is None or len(points) < 3:
        return points
    c = points.mean(axis=0)
    e1, e2 = plane_basis(n)
    uv = np.stack([np.dot(points - c, e1), np.dot(points - c, e2)], axis=1)
    ang = np.arctan2(uv[:,1], uv[:,0])
    return points[np.argsort(ang)]

def plane_from_point_normal(p0, n):
    n = normalize(n)
    d = float(np.dot(n, p0))
    return n, d

# =========================================================
# Roof model / Frames
# =========================================================
def roof_normals(m1, m2):
    # plane1: z = m1 x => n=(m1,0,-1)
    # plane2: z = m2 y => n=(0,m2,-1)
    n1 = normalize(np.array([m1, 0.0, -1.0], dtype=float))
    n2 = normalize(np.array([0.0, m2, -1.0], dtype=float))
    return n1, n2

def hip_frame(m1, m2):
    # intersection direction of two planes
    v = np.array([m2, m1, m1*m2], dtype=float)
    xh = normalize(v)
    zh = normalize(Zhat - np.dot(Zhat, xh)*xh)  # plumb (no roll)
    yh = normalize(np.cross(zh, xh))
    return make_R_from_axes(xh, yh, zh)

def rafter1_frame(m1):
    rx = normalize(np.array([1.0, 0.0, m1], dtype=float))     # run direction
    rz = normalize(np.array([-m1, 0.0, 1.0], dtype=float))    # top normal
    ry = normalize(np.cross(rz, rx))                          # RIGHT is +y
    return make_R_from_axes(rx, ry, rz)

def rafter2_frame(m2):
    rx = normalize(np.array([0.0, 1.0, m2], dtype=float))
    rz = normalize(np.array([0.0, -m2, 1.0], dtype=float))
    ry = normalize(np.cross(rz, rx))
    return make_R_from_axes(rx, ry, rz)

# =========================================================
# UNFOLD: face classification
# =========================================================
def face_of_segment_by_endpoints(a_l: np.ndarray, b_l: np.ndarray, w: float, h: float) -> str:
    yw, zh = w/2.0, h/2.0
    scores = {
        'RIGHT': abs(a_l[1]-yw) + abs(b_l[1]-yw),
        'LEFT':  abs(a_l[1]+yw) + abs(b_l[1]+yw),
        'TOP':   abs(a_l[2]-zh) + abs(b_l[2]-zh),
        'BOTTOM':abs(a_l[2]+zh) + abs(b_l[2]+zh),
    }
    return min(scores, key=scores.get)

def to2d(face: str, p_local: np.ndarray, w: float, h: float) -> np.ndarray:
    x,y,z = p_local
    yw = w/2.0
    zh = h/2.0

    if face == 'TOP':
        return np.array([x, y + yw], dtype=float)
    if face == 'BOTTOM':
        return np.array([x, yw - y], dtype=float)
    if face == 'RIGHT':
        return np.array([x, z + zh], dtype=float)
    if face == 'LEFT':
        return np.array([x, zh - z], dtype=float)

    raise ValueError(face)

def select_main_segment(segments_2d):
    if not segments_2d:
        return None
    best = None
    best_key = None
    for a,b in segments_2d:
        mid = 0.5*(a+b)
        x0 = abs(float(mid[0]))
        length = float(np.linalg.norm(b-a))
        key = (x0, -length)
        if best_key is None or key < best_key:
            best_key = key
            best = (a,b)
    return best

def draw_face(ax, face: str, w: float, h: float, L: float, segments_2d, title: str, view_span: float, ink_color: str):
    ax.set_title(title)
    # 横長潰れを防ぐ（前回寄せ）：equal をやめる
    ax.set_aspect('auto')
    ax.grid(False)

    x0, x1 = -L/2.0, L/2.0

    if face in ('RIGHT','LEFT'):
        v0, v1 = 0.0, h
        member_v = h
        ax.set_xlabel('x')
        ax.set_ylabel('z (unfold)')
    else:
        v0, v1 = 0.0, w
        member_v = w
        ax.set_xlabel('x')
        ax.set_ylabel('y (unfold)')

    rect = np.array([[x0,v0],[x1,v0],[x1,v1],[x0,v1],[x0,v0]])
    ax.plot(rect[:,0], rect[:,1], linewidth=1, color='#777777')

    for a,b in segments_2d:
        ax.plot([a[0],b[0]],[a[1],b[1]], linewidth=3, color=ink_color)

    main = select_main_segment(segments_2d)
    if main is None:
        ax.set_xlim(-250,250)
        ax.set_ylim(-0.4*member_v, member_v + 0.9*member_v)
        return

    a,b = main
    mid = 0.5*(a+b)

    dx = abs(b[0]-a[0])
    dv = abs(b[1]-a[1])

    tol = 0.5
    show_dx = dx > tol
    show_dv = (dv > tol) and (abs(dv - member_v) >= tol)

    vals = []
    if show_dx: vals.append(f'{dx:.2f}')
    if show_dv: vals.append(f'{dv:.2f}')

    halfx = view_span/2.0
    ax.set_xlim(mid[0]-halfx, mid[0]+halfx)

    pad_v = 0.80 * member_v
    ax.set_ylim(-pad_v, member_v + pad_v)

    if vals and show_dx:
        yD = v0 - 0.28*member_v
        xA = min(a[0], b[0])
        xB = max(a[0], b[0])

        ax.plot([xA, xA], [v0, yD], color='red', linewidth=1.2)
        ax.plot([xB, xB], [v0, yD], color='red', linewidth=1.2)
        ax.annotate('', xy=(xB, yD), xytext=(xA, yD),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.6))

        ax.text(0.5*(xA+xB), yD - 0.14*member_v, ' / '.join(vals),
                ha='center', va='top', fontsize=11, color='black',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

# =========================================================
# Core: cut prism by arbitrary plane, then unfold segments
# =========================================================
def cut_and_unfold(w, h, R_member, plane_n, plane_d, L=2000.0):
    V = build_prism_vertices(w, h, L)
    VW = (R_member @ V.T).T

    pts = []
    for i,j in BOX_EDGES:
        p = intersect_segment_plane(VW[i], VW[j], plane_n, plane_d)
        if p is not None:
            pts.append(p)

    P = unique_points(pts, tol=1e-6)
    if P is None or len(P) < 3:
        return None, {'TOP':[], 'BOTTOM':[], 'LEFT':[], 'RIGHT':[]}

    P = order_polygon(P, plane_n)
    Pin_local = (R_member.T @ P.T).T

    face_segments = {'TOP':[], 'BOTTOM':[], 'LEFT':[], 'RIGHT':[]}
    for k in range(len(Pin_local)):
        a_l = Pin_local[k]
        b_l = Pin_local[(k+1) % len(Pin_local)]
        face = face_of_segment_by_endpoints(a_l, b_l, w, h)
        face_segments[face].append((to2d(face, a_l, w, h), to2d(face, b_l, w, h)))

    return P, face_segments

# =========================================================
# HIP: true section (local YZ) with 2 roof lines
# =========================================================
def clip_polygon_halfplane(poly, ny, nz, keep_sign):
    if poly is None or len(poly) < 3:
        return poly
    out = []
    eps = 1e-12

    def f(P):
        return ny*P[0] + nz*P[1]

    def inside(P):
        val = f(P)
        if keep_sign > 0:
            return val >= -1e-9
        else:
            return val <= +1e-9

    def intersect(A,B):
        fA = f(A); fB = f(B)
        denom = (fA - fB)
        if abs(denom) < eps:
            return None
        t = fA / (fA - fB)
        return A + t*(B-A)

    n = len(poly)
    for i in range(n):
        A = poly[i]
        B = poly[(i+1) % n]
        Ain = inside(A)
        Bin = inside(B)

        if Ain and Bin:
            out.append(B)
        elif Ain and (not Bin):
            I = intersect(A,B)
            if I is not None:
                out.append(I)
        elif (not Ain) and Bin:
            I = intersect(A,B)
            if I is not None:
                out.append(I)
            out.append(B)

    if len(out) < 3:
        return None

    clean = []
    for p in out:
        if len(clean)==0 or np.linalg.norm(p-clean[-1]) > 1e-6:
            clean.append(p)
    if len(clean) >= 2 and np.linalg.norm(clean[0]-clean[-1]) < 1e-6:
        clean.pop()
    return np.array(clean, dtype=float)

def segment_of_line_inside_polygon(poly, ny, nz):
    if poly is None or len(poly) < 3:
        return None
    pts = []
    n = len(poly)
    eps = 1e-12

    def f(P):
        return ny*P[0] + nz*P[1]

    for i in range(n):
        A = poly[i]
        B = poly[(i+1)%n]
        fA = f(A); fB = f(B)

        if abs(fA) < 1e-9:
            pts.append(A.copy())
        if fA*fB < 0:
            denom = (fA - fB)
            if abs(denom) > eps:
                t = fA/(fA-fB)
                I = A + t*(B-A)
                pts.append(I)

    uniq=[]
    for p in pts:
        if all(np.linalg.norm(p-q) > 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None

    P = np.array(uniq, dtype=float)
    bi,bj,bd=0,1,-1.0
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            d=float(np.linalg.norm(P[i]-P[j]))
            if d>bd:
                bd=d; bi=i; bj=j
    return P[bi], P[bj]

def draw_dim_vertical_zero_to(ax, y, z_end, text, color='red'):
    # 寸法：Z=0 から 切り墨端部(z_end) まで
    z0 = 0.0
    z1 = z_end
    ax.plot([y,y],[z0,z1], color=color, linewidth=1.3)
    ax.annotate('', xy=(y, z1), xytext=(y, z0),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.6))
    ax.text(y+2.0, 0.5*(z0+z1), text,
            ha='left', va='center', rotation=90,
            fontsize=11, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

def draw_hip_true_section(ax, Rhip, a, b, n1, n2):
    ax.set_title('HIP true section (wood end, local YZ)  ※山形断面')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlabel('Y (mm)   (RIGHT=+ / LEFT=-)')
    ax.set_ylabel('Z (mm)')

    y0,y1 = -a/2.0, +a/2.0
    z0,z1 = -2b, 0.0

    rect = np.array([[y0,z0],[y1,z0],[y1,z1],[y0,z1]], dtype=float)

    n1_loc = Rhip.T @ n1
    n2_loc = Rhip.T @ n2
    ny1,nz1 = float(n1_loc[1]), float(n1_loc[2])
    ny2,nz2 = float(n2_loc[1]), float(n2_loc[2])

    test = np.array([0.0, z0], dtype=float)
    s1 = ny1*test[0] + nz1*test[1]
    s2 = ny2*test[0] + nz2*test[1]
    keep1 = +1 if s1 >= 0 else -1
    keep2 = +1 if s2 >= 0 else -1

    poly = rect.copy()
    poly = clip_polygon_halfplane(poly, ny1, nz1, keep1)
    poly = clip_polygon_halfplane(poly, ny2, nz2, keep2)

    if poly is None:
        rc = np.vstack([rect, rect[0]])
        ax.plot(rc[:,0], rc[:,1], color='#777777', linewidth=1.2)
        ax.text(0.5,0.5,'NO CLIP', transform=ax.transAxes, ha='center', va='center')
        return 0.0, 0.0

    pc = np.vstack([poly, poly[0]])
    ax.fill(poly[:,0], poly[:,1], alpha=0.08)
    ax.plot(pc[:,0], pc[:,1], color='#444444', linewidth=2.0)

    seg1 = segment_of_line_inside_polygon(poly, ny1, nz1)
    seg2 = segment_of_line_inside_polygon(poly, ny2, nz2)
    if seg1 is not None:
        A,B = seg1
        ax.plot([A[0],B[0]],[A[1],B[1]], color='orange', linewidth=3)
    if seg2 is not None:
        A,B = seg2
        ax.plot([A[0],B[0]],[A[1],B[1]], color='dodgerblue', linewidth=3)

    ax.plot([0],[0], marker='o', markersize=4, color='black')

    def yz_tan(ny,nz):
        if abs(nz) < 1e-12:
            return float('inf')
        return -(ny/nz)

    tan1 = yz_tan(ny1,nz1)
    tan2 = yz_tan(ny2,nz2)

    def z_on_line_at_y(ny, nz, yy):
        if abs(nz) < 1e-12:
            return None
        return -(ny/nz) * yy

    # 端部の切り墨 z（Z=0 からの距離を出す）
    yR = y1
    zR = z_on_line_at_y(ny1, nz1, yR)

    yL = y0
    zL = z_on_line_at_y(ny2, nz2, yL)

    if zR is not None:
        cutR = abs(zR - 0.0)
        draw_dim_vertical_zero_to(ax, yR + 6.0, zR, f'cut(R)= {cutR:.2f}mm')

    if zL is not None:
        cutL = abs(zL - 0.0)
        draw_dim_vertical_zero_to(ax, yL - 6.0, zL, f'cut(L)= {cutL:.2f}mm')

    m = 0.10*max(a,b) + 10.0
    ax.set_xlim(y0-m, y1+m)
    ax.set_ylim(z0-m, z1+m)

    return tan1, tan2

# =========================================================
# PLAN (2D): RAFTER1 / RAFTER2 / HIP
# =========================================================
def hip_slope_tan(m1, m2):
    return (m1*m2) / math.sqrt(m1*m1 + m2*m2)

def hip_plan_angle_deg(m1, m2):
    return math.degrees(math.atan2(m1, m2))

# =========================
# PLAN (2D)  ※差し替え用（HIP文字だけ残す）
# (0,0)=隅木×垂木 交点
# RAFTER1 は -X 方向へ（HIP端点まで）
# RAFTER2 は -Y 方向へ（HIP端点まで）
# HIP は (-,-) 象限へ
# =========================
def draw_plan_2d(ax, m1, m2):
    ax.set_title('PLAN (2D): RAFTER1 / RAFTER2 / HIP')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    theta = hip_plan_angle_deg(m1, m2)  # deg
    th = math.radians(theta)

    # 表示スケール（任意）
    Lh = 115.0
    pad = 18.0

    # ---- HIP direction: (-cos, -sin) ----
    hx = -Lh * math.cos(th)
    hy = -Lh * math.sin(th)
    ax.plot([0, hx], [0, hy], color='magenta', linewidth=5)

    # ---- RAFTER length: HIP端点レベルに合わせる ----
    ax.plot([0, hx], [0, 0], color='black', linewidth=3)   # RAFTER1 (-X)
    ax.plot([0, 0], [0, hy], color='black', linewidth=3)   # RAFTER2 (-Y)

    # ---- labels ----
    ax.text(0.5*hx, +5, 'RAFTER1', color='black', fontsize=14,
            ha='center', va='bottom')
    ax.text(+5, 0.5*hy, 'RAFTER2', color='black', fontsize=14,
            ha='left', va='center', rotation=90)

    # HIP ラベル（上下逆さまにならない角度に丸める）
    hip_angle = math.degrees(math.atan2(hy, hx))
    if hip_angle < -90:
        hip_angle += 180
    elif hip_angle > 90:
        hip_angle -= 180

    ax.text(0.55*hx, 0.55*hy, 'HIP',
            color='black', fontsize=14,
            ha='center', va='center', rotation=hip_angle)

    # ---- view ----
    xmin = min(hx, 0.0) - pad
    ymin = min(hy, 0.0) - pad
    xmax = 0.0 + pad
    ymax = 0.0 + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)




# =========================================================
# Render
# =========================================================
def render_png_b64(m1, m2, a, b, rw, rh, view_span):
    Rhip = hip_frame(m1, m2)
    Rr1  = rafter1_frame(m1)
    Rr2  = rafter2_frame(m2)

    n1, n2 = roof_normals(m1, m2)

    plane_roof1_n, plane_roof1_d = plane_from_point_normal(np.array([0.0, 0.0, 0.0]), n1)
    plane_roof2_n, plane_roof2_d = plane_from_point_normal(np.array([0.0, 0.0, 0.0]), n2)

    # hip side planes in world (y = +/- a/2 in hip-local)
    n_side_plus = Rhip @ np.array([0.0, +1.0, 0.0], dtype=float)
    p_side_plus = Rhip @ np.array([0.0, +a/2.0, 0.0], dtype=float)
    plane_side_plus_n, plane_side_plus_d = plane_from_point_normal(p_side_plus, n_side_plus)

    n_side_minus = Rhip @ np.array([0.0, -1.0, 0.0], dtype=float)
    p_side_minus = Rhip @ np.array([0.0, -a/2.0, 0.0], dtype=float)
    plane_side_minus_n, plane_side_minus_d = plane_from_point_normal(p_side_minus, n_side_minus)

    # RAFTERS: cut by hip side planes
    _, segs_r1 = cut_and_unfold(rw, rh, Rr1, plane_side_plus_n, plane_side_plus_d)
    _, segs_r2 = cut_and_unfold(rw, rh, Rr2, plane_side_minus_n, plane_side_minus_d)

    # Figure layout
    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.20, 1.0, 1.0])

    # ---- Row1: 2 columns
    #   left : HIP true section
    #   right: PLAN (top) + PARAM (bottom)
    row1 = gs[0, 0].subgridspec(1, 2, width_ratios=[1.45, 1.10], wspace=0.25)

    ax_hip = fig.add_subplot(row1[0, 0])

    right = row1[0, 1].subgridspec(2, 1, height_ratios=[1.00, 0.42], hspace=0.06)
    ax_plan = fig.add_subplot(right[0, 0])
    ax_txt  = fig.add_subplot(right[1, 0])
    ax_txt.axis('off')

    hip_tan1, hip_tan2 = draw_hip_true_section(ax_hip, Rhip, a, b, n1, n2)
    draw_plan_2d(ax_plan, m1, m2)

    mhip  = hip_slope_tan(m1, m2)
    theta = hip_plan_angle_deg(m1, m2)
    plan_tan = math.tan(math.radians(theta))

    lines = []
    lines.append('      ')
    lines.append('      ')
    lines.append('      ')
    lines.append('HIP')
    lines.append(f'  plan angle tan = {plan_tan:.6f}')
    lines.append(f'  flow slope tan = {mhip:.6f}')
    lines.append(f'  bevel tan (P1 on YZ) = {hip_tan1:.6f}')
    lines.append(f'  bevel tan (P2 on YZ) = {hip_tan2:.6f}')

    ax_txt.text(
        0.0, 1.0,
        '\\n'.join(lines),
        ha='left', va='top',
        fontsize=11, family='monospace'
    )

    # ---- helper: one unfold row (4 panels)
    def unfold_row(ax_row, segs, title, W, H, ink='orange'):
        ax_row.axis('off')
        ax_row.text(
            0.0, 1.02, title,
            transform=ax_row.transAxes,
            ha='left', va='bottom',
            fontsize=14
        )

        x0, y0, w0, h0 = 0.02, 0.06, 0.96, 0.88
        gap_x = 0.06
        gap_y = 0.16

        Wp = (w0 - gap_x) / 2.0
        Hp = (h0 - gap_y) / 2.0

        ax_top    = ax_row.inset_axes([x0,               y0 + Hp + gap_y, Wp, Hp])
        ax_right  = ax_row.inset_axes([x0 + Wp + gap_x,  y0 + Hp + gap_y, Wp, Hp])
        ax_left   = ax_row.inset_axes([x0,               y0,              Wp, Hp])
        ax_bottom = ax_row.inset_axes([x0 + Wp + gap_x,  y0,              Wp, Hp])

        draw_face(ax_top,    'TOP',    W, H, 2000.0, segs.get('TOP', []),    'TOP',    view_span, ink)
        draw_face(ax_right,  'RIGHT',  W, H, 2000.0, segs.get('RIGHT', []),  'RIGHT',  view_span, ink)
        draw_face(ax_left,   'LEFT',   W, H, 2000.0, segs.get('LEFT', []),   'LEFT',   view_span, ink)
        draw_face(ax_bottom, 'BOTTOM', W, H, 2000.0, segs.get('BOTTOM', []), 'BOTTOM', view_span, ink)

    ax_r1 = fig.add_subplot(gs[1, 0])
    unfold_row(ax_r1, segs_r1, 'RAFTER1 UNFOLD (cut by HIP side +a/2)', rw, rh, ink='orange')

    ax_r2 = fig.add_subplot(gs[2, 0])
    unfold_row(ax_r2, segs_r2, 'RAFTER2 UNFOLD (cut by HIP side -a/2)', rw, rh, ink='orange')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

`;

function applyQueryParamsToInputs() {
  const p = new URLSearchParams(location.search);
  const keys = ["m1","m2","a","b","rw","rh","span"];
  for (const k of keys) {
    if (p.has(k)) document.getElementById(k).value = p.get(k);
  }
}

function getNum(id) { return Number(document.getElementById(id).value); }

async function init() {
  const status = document.getElementById("status");
  try {
    status.textContent = "Loading Pyodide…";
    pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/"
    });

    status.textContent = "Loading numpy/matplotlib…";
    await pyodide.loadPackage(["numpy", "matplotlib"]);

    status.textContent = "Loading Python code…";
    await pyodide.runPythonAsync(PY_CODE);

    status.textContent = "Ready.";
  } catch (e) {
    console.error("INIT FAILED:", e);
    status.textContent = "PY_CODE ERROR: " + e;
    throw e;
  }
}



async function draw() {
  const status = document.getElementById("status");
  const img = document.getElementById("img");
  try {
    status.textContent = "Rendering…";

    let m1 = getNum("m1");
    let m2 = getNum("m2");
    const a = getNum("a");
    const b = getNum("b");
    const rw = getNum("rw");
    const rh = getNum("rh");
    const span = getNum("span");

    document.getElementById("m1").value = m1.toFixed(6);
    document.getElementById("m2").value = m2.toFixed(6);

    pyodide.globals.set("M1_IN", m1);
    pyodide.globals.set("M2_IN", m2);
    pyodide.globals.set("A_IN", a);
    pyodide.globals.set("B_IN", b);
    pyodide.globals.set("RW_IN", rw);
    pyodide.globals.set("RH_IN", rh);
    pyodide.globals.set("SPAN_IN", span);

    const b64 = await pyodide.runPythonAsync(
      "render_png_b64(M1_IN, M2_IN, A_IN, B_IN, RW_IN, RH_IN, SPAN_IN)"
    );

    img.src = "data:image/png;base64," + b64;
    status.textContent = "Done.";
  } catch (e) {
    console.error(e);
    status.textContent = "Error: " + e;
  }
}

document.getElementById("run").addEventListener("click", draw);

applyQueryParamsToInputs();
init().then(draw);
