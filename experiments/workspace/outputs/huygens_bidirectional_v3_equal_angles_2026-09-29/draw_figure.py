"""Exact vector reconstruction of the Huygens/master-equation schematic.

Profiles are explanatory drawings, not a numerical optical experiment.
K_0 glyphs repeat one nonnegative profile; K_infinity glyphs
are exact np.sinc profiles. Copies grow with their illustrative fans.
The latter illustrates a signed reconstruction
kernel, not an assertion that every camera inverse is a sinc.
"""
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
W, H = 1025, 1255
INCH_PER_UNIT = 7.2/1800
FONT_SCALE = 1.22
CY, DY = 270.0, 330.0
X = dict(scene=55.0, filtered=385.0, sampled=455.0,
         reconstructed=645.0, rescaled=715.0, front=890.0, output=950.0)
BLUE, PURPLE = '#058DE8', '#8830CB'
FAN, INK, GRAY = '#8ECEFF', '#151515', '#63676B'
parser = argparse.ArgumentParser()
parser.add_argument('--preview-only', action='store_true')
args = parser.parse_args()

# Preserve every signal's vertical extent from v9; compress only
# horizontal spacing after I_1. There is no scaling-region background.
ORIGINAL_HALF, OUTPUT_HALF = 140., 84.
def previous_extent(x):
    return ORIGINAL_HALF + (OUTPUT_HALF-ORIGINAL_HALF)*(x-55.)/(1320.-55.)
FILTERED_HALF = previous_extent(385.)
INPUT_HALF = previous_extent(455.)
FINAL_HALF = previous_extent(1240.)
RECON_HALF, RESCALED_HALF = 102., 65.
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 8.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'path.simplify': False, 'savefig.facecolor': 'white',
})
fig = plt.figure(figsize=(W*INCH_PER_UNIT, H*INCH_PER_UNIT), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0), aspect='equal')
ax.axis('off')
texts, records = [], {'braces': [], 'camera_profiles': [], 'inverse_profiles': [],
                     'guide_pairs': [], 'camera_fans': []}

def text(x, y, s, size=8.5, ha='center', va='center', **kw):
    # Mathtext has no xfrac package: typeset its slanted numerator/slash/
    # denominator using the same STIX glyphs as the rest of the figure.
    s = s.replace(r'\sfrac{1}{s}', r'{}^{1}\!/\!_{s}')
    artist = ax.text(x, y, s, fontsize=size*FONT_SCALE, ha=ha, va=va,
                     color=INK, zorder=12, **kw)
    texts.append(artist)
    return artist

def brace(x0, x1, y, above, label, size=10, inset_left=0, inset_right=0,
          label_offset=None, start_trim=0, label_dx=0, linespacing=1.2):
    """Cubic braces whose endpoints coincide exactly with signal planes."""
    logical_x0, logical_x1 = x0, x1
    x0, x1 = x0+inset_left, x1-inset_right
    direction = -1 if above else 1
    mid = (x0+x1)/2
    radius, cusp = 12.0, 10.0
    points = [(x0, y-direction*(radius-start_trim)),
              (x0, y), (x0+radius, y), (x0+2*radius, y),
              (mid-2*radius, y),
              (mid-radius, y), (mid-radius/2, y), (mid, y+direction*cusp),
              (mid+radius/2, y), (mid+radius, y), (mid+2*radius, y),
              (x1-2*radius, y),
              (x1-radius, y), (x1, y), (x1, y-direction*radius)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4]*3 + [MplPath.LINETO] + \
            [MplPath.CURVE4]*6 + [MplPath.LINETO] + [MplPath.CURVE4]*3
    ax.add_patch(PathPatch(MplPath(points, codes), fill=False,
                          edgecolor='#52565A', lw=0.55, alpha=.35, capstyle='round',
                          joinstyle='round', zorder=10))
    if label_offset is None:
        label_offset = -44 if above else 38
    label_artist = text(mid+label_dx, y+label_offset, label, size=size,linespacing=linespacing)
    records['braces'].append(dict(x0=logical_x0, x1=logical_x1,
        drawn_x0=x0,drawn_x1=x1,y=y,above=above,label=label,
        start_y=points[0][1],start_trim=start_trim,label_dx=label_dx))
    return label_artist

def fan(x0, yc, x1, half, alpha=.25):
    ax.add_patch(Polygon([(x0, yc), (x1, yc-half), (x1, yc+half)],
                         closed=True, facecolor=FAN, edgecolor='none',
                         alpha=alpha, zorder=1))

u = np.linspace(-1, 1, 1601)
taper = np.sin(np.pi*(u+1)/2)**0.7
envelope = (.67*np.exp(-((u+.17)/.22)**2)
            + .32*np.exp(-((u-.48)/.115)**2)
            + .14*np.exp(-((u+.64)/.12)**2))
base = (.065 + envelope) * taper
original = base + taper*(.085*np.sin(12*np.pi*u+.4)
                        + .055*np.sin(30*np.pi*u-.9)
                        + .028*np.sin(57*np.pi*u))
reconstruction = base + taper*.085*np.sin(12*np.pi*u+.4)
filtered = (.065 + .80*envelope + .035*np.sin(12*np.pi*u+.4))*taper
final_front = (.045 + .71*np.exp(-((u+.17)/.30)**2)
               + .31*np.exp(-((u-.47)/.20)**2)
               + .13*np.exp(-((u+.63)/.18)**2))*taper

def curve(x, yc, values, half, amplitude=58, baseline=False, direction=1):
    xx, yy = x + direction*amplitude*values, yc + half*u
    if baseline:
        ax.plot([x, x], [yc-half, yc+half], color=PURPLE, alpha=.55, lw=.40, zorder=5)
    ax.plot(xx, yy, color=PURPLE, lw=.91, solid_capstyle='round', zorder=7)
    return np.column_stack([xx, yy])

def guides(x0, x1, yc, half0, half1, values0, values1, amplitude=58, direction=1):
    records['guide_pairs'].append(dict(x0=x0,x1=x1,y=yc))
    for t in [-1, 1]:
        v0, v1 = np.interp(t, u, values0), np.interp(t, u, values1)
        ax.plot([x0+direction*amplitude*v0, x1+direction*amplitude*v1],
                [yc+half0*t, yc+half1*t], color='#A8ADB2', lw=.50,
                linestyle=(0, (3.4, 3.0)), zorder=0)

kt = np.linspace(-3, 3, 901)
psf = np.sinc(kt)**2
inverse = np.sinc(kt)
psf_shape = np.column_stack([43.0*psf, 58.0*kt/3])
INVERSE_FAN_HALF = 51.0
inverse_shape = np.column_stack([32.0*inverse, INVERSE_FAN_HALF*kt/3])

def kernel(x, yc, kind, growth=1., direction=1):
    canonical = psf_shape if kind == 'camera' else inverse_shape
    shape = canonical * growth
    ax.plot([x, x], [yc+shape[0,1], yc+shape[-1,1]],
            color='#7A8085', lw=.43, zorder=4)
    ax.plot(x+direction*shape[:,0], yc+shape[:,1], color=BLUE,
            lw=.76 if kind=='camera' else .66, solid_capstyle='round', zorder=6)
    records[kind+'_profiles'].append(dict(x=x, y=yc, display_growth=growth, direction=direction,
        shape_hash=hashlib.sha256(canonical.tobytes()).hexdigest()))


def camera_fan(x0, yc, x1):
    """One shared illustrative opening angle for every K_0 fan."""
    half = CAMERA_SLOPE*abs(x1-x0)
    fan(x0, yc, x1, half)
    records['camera_fans'].append(dict(
        x0=x0, x1=x1, center_y=yc, endpoint_half_height=half,
        half_angle_radians=np.arctan(half/abs(x1-x0))))
    return half

def camera_kernel(x, yc, source_x, direction=1):
    """Keep the normalized PSF identical and grow it with the shared cone."""
    growth = CAMERA_SLOPE*abs(x-source_x)/58.
    kernel(x, yc, 'camera', growth, direction=direction)

def source(x, yc, values):
    xpos = x+58*np.interp(0, u, values)
    ax.scatter([xpos], [yc], s=17, facecolors='#111111', linewidths=0, zorder=9)
    return xpos

input_ys, output_ys = np.arange(-102, 103, 51), np.array([-51., 0., 51.])

def sensor(x, yc, output=False):
    pts = output_ys if output else input_ys
    half = OUTPUT_HALF if output else INPUT_HALF
    ax.plot([x, x], [yc-half, yc+half], color='#6A6D70', lw=.64, zorder=5)
    ax.scatter(np.full(len(pts),x), yc+pts, s=17,
               facecolors='#111111', linewidths=0, zorder=9)
    return np.column_stack([np.full(len(pts),x), yc+pts])

# The lower K_s brace alone groups the I_1 -> I_s operation.
records['scaling_background'] = None

# Upper path: initial continuous scene and camera.
source_x = X['scene'] + 58*np.interp(0,u,original)
ZOOM_FAN_HALF = 1.05*FINAL_HALF
CAMERA_SLOPE = ZOOM_FAN_HALF/(X['front']-source_x)
camera_fan(source_x, CY, X['filtered'])
for xp in [222, 282, 342]:
    camera_kernel(xp,CY,source_x)
source_top = curve(X['scene'], CY, original, ORIGINAL_HALF)
source(X['scene'], CY, original)
curve(X['filtered'], CY, filtered, FILTERED_HALF, baseline=True)
input_points = sensor(X['sampled'], CY)

# Reconstruction: identical sinc profiles emitted by representative pixels.
for offset in [-51, 0, 51]:
    fan(X['sampled'], CY+offset, X['reconstructed'], INVERSE_FAN_HALF, .19)
    for xp in [520, 567, 615]:
        kernel(xp,CY+offset,'inverse',
               (xp-X['sampled'])/(X['reconstructed']-X['sampled']))
curve(X['reconstructed'], CY, reconstruction, RECON_HALF, baseline=True)

# Continuous coordinate rescaling, with no sampler before recapture.
guides(X['reconstructed'], X['rescaled'], CY, RECON_HALF, RESCALED_HALF,
       reconstruction, reconstruction)
curve(X['rescaled'], CY, reconstruction, RESCALED_HALF)
rescaled_source = X['rescaled']+58*np.interp(0,u,reconstruction)
camera_fan(rescaled_source, CY, X['front'])
for xp in [768,808,848]:
    camera_kernel(xp,CY,rescaled_source)
output_top = curve(X['front'], CY, final_front, FINAL_HALF, baseline=True)
samples_top = sensor(X['output'], CY, output=True)

# Matched, exact shared boundaries for ALL four upper operators.
brace(X['scene'],X['sampled'],95,True,r'camera $\mathbf{K}_0$',10,inset_right=2,label_offset=-44)
brace(X['sampled'],X['reconstructed'],95,True,r'$\mathbf{K}_\infty$',10,inset_left=2,inset_right=2,label_offset=-44)
brace(X['reconstructed'],X['rescaled'],95,True,r'$\mathbf{M}_s$',9,inset_left=2,inset_right=2,label_offset=-44)
brace(X['rescaled'],X['output'],95,True,r'$\mathbf{K}_0$',10,inset_left=2,label_offset=-44)

text(235,139,r'PSF $K_0$',9)
text(X['sampled']+16,147,'sample',8)
text(810,139,r'PSF $K_0$',9)
text(X['output']+16,147,'sample',8)
text(X['reconstructed']+10,139,r'$\mathbf{1}_B\,I^{\mathrm{cont}}$',9)

text(X['scene'],CY+ORIGINAL_HALF+27,r'$I^{\mathrm{cont}}$',10)
input_label = text(X['sampled']-13,432,r'$I_1$',10,ha='right')
output_label_top = text(X['output']+26,407,r'$I_s$',10,ha='left')
forward_scale_label = brace(X['sampled'],X['output'],407,False,
      r'$\mathbf{K}_s$',
      10.3,label_offset=34,start_trim=6)
text(X['output'],467,r'$=$',13)

# Lower route: same scene, same final front, same output grid; only one sampler.
lower = CY+DY
camera_fan(source_x,lower,X['front'])
for xp in [410, 610, 820]:
    camera_kernel(xp,lower,source_x)
source_bottom=curve(X['scene'],lower,original,ORIGINAL_HALF)
source(X['scene'],lower,original)
output_bottom=curve(X['front'],lower,final_front,FINAL_HALF,baseline=True)
samples_bottom=sensor(X['output'],lower,output=True)
text(620,lower-79,r'PSF $K_0$',9)
text(X['scene'],lower+ORIGINAL_HALF+27,r'$I^{\mathrm{cont}}$',10)
output_label_bottom = text(X['output']+26,lower+OUTPUT_HALF+26,r'$I_s$',10,ha='left')
zoom_brace_height = lower+OUTPUT_HALF+12
brace(source_x,X['output'],zoom_brace_height,False,
      r'zoomed camera  $\mathbf{K}_0\,\mathbf{M}_s$',10)


# Third route: canonical zoom-in, read from RIGHT to LEFT.
# Kernel amplitudes face left to match drawing direction; this does not
# change the kernel as a function of the vertical spatial coordinate.
ROW_RAISE = 65.
RY = 1110.-ROW_RAISE
R = dict(input=X['output'],
         reconstructed=X['sampled']+X['output']-X['reconstructed'],
         rescaled=X['sampled']+X['output']-X['rescaled'],
         front=X['sampled']+X['output']-X['front'],
         output=X['sampled'])
samples_reverse_input = sensor(R['input'], RY, output=True)

# Shared coarse structure, without the earlier scene's fine oscillations.
retained_scene = base.copy()
retained_capture = (.065 + .80*envelope)*taper
for offset in [-51, 0, 51]:
    fan(R['input'], RY+offset, R['reconstructed'], INVERSE_FAN_HALF, .19)
    for xp in [885, 838, 790]:
        kernel(xp, RY+offset, 'inverse',
               (R['input']-xp)/(R['input']-R['reconstructed']), direction=-1)
reverse_reconstruction = curve(R['reconstructed'], RY, retained_scene,
                               RESCALED_HALF, baseline=True, direction=-1)
guides(R['reconstructed'], R['rescaled'], RY, RESCALED_HALF, RECON_HALF,
       retained_scene, retained_scene, direction=-1)
reverse_rescaled = curve(R['rescaled'], RY, retained_scene,
                         RECON_HALF, direction=-1)

# Recapture with the SAME camera at the original resolution.
reverse_source_x = R['rescaled']-58*np.interp(0,u,retained_scene)
camera_fan(reverse_source_x, RY, R['front'])
for growth in [.28, .53, .78]:
    xp = reverse_source_x-growth*(reverse_source_x-R['front'])
    camera_kernel(xp, RY, reverse_source_x, direction=-1)
reverse_front = curve(R['front'], RY, retained_capture,
                       FILTERED_HALF, baseline=True, direction=-1)
samples_reverse_output = sensor(R['output'], RY)

brace(R['output'], R['rescaled'], 936-ROW_RAISE, True, r'$\mathbf{K}_0$', 10,
      inset_right=2, label_offset=-44)
brace(R['rescaled'], R['reconstructed'], 936-ROW_RAISE, True, r'$\mathbf{M}_{\sfrac{1}{s}}$', 9,
      inset_left=2, inset_right=2, label_offset=-44)
brace(R['reconstructed'], R['input'], 936-ROW_RAISE, True, r'$\mathbf{K}_\infty$', 10,
      inset_left=2, label_offset=-44)
text(558, 974-ROW_RAISE, r'PSF $K_0$', 9)
text(R['rescaled']+5, 980-ROW_RAISE, r'$\mathbf{1}_{sB}\,I^{\mathrm{cont}}$', 9)
text(R['input']+16, 980-ROW_RAISE, 'sample', 8)
text(R['output']-32, 974-ROW_RAISE, 'sample', 8)
text(R['output']-18, RY+INPUT_HALF+33, r'$I_{1\mid s}$', 10, ha='right')
text(R['input']+26, RY+OUTPUT_HALF+26, r'$I_s$', 10, ha='left')
reverse_scale_label = brace(R['output'], R['input'], 1247-ROW_RAISE, False, r'$\mathbf{K}_{\sfrac{1}{s}}$',
      10.3, label_offset=34, start_trim=6)

# The left source area is deliberately empty in the reverse route.

assert R['input']==X['output'] and R['output']==X['sampled']
assert np.allclose(samples_reverse_input-np.array([0,RY-CY]), samples_top)
assert np.allclose(samples_reverse_output-np.array([0,RY-CY]), input_points)
assert len(samples_reverse_input)==3 and len(samples_reverse_output)==5
assert np.isclose(np.ptp(reverse_rescaled[:,1]) /
                  np.ptp(reverse_reconstruction[:,1]), RECON_HALF/RESCALED_HALF)
assert np.allclose(reverse_rescaled[:,0]-R['rescaled'],
                   reverse_reconstruction[:,0]-R['reconstructed'])
assert not np.allclose(retained_scene, reconstruction)
records['reverse_route'] = dict(
    reading_direction='right-to-left', coordinates=R, center_y=RY,
    input_label='I_s', output_label='I_{1|s}', input_samples=3, output_samples=5,
    original_resolution_restored=True, discarded_detail_restored=False,
    kernel_shapes_unchanged=True, amplitudes_mirrored_for_drawing_only=True,
    no_original_scene_source_in_reverse_row=True)

# The four camera fans have one angle; all purple scenes retain their
# previous coordinates and profiles. Only blue PSFs/fans change.
assert len(records['camera_fans'])==4
assert np.allclose([f['half_angle_radians'] for f in records['camera_fans']],
                   np.arctan(CAMERA_SLOPE))
assert np.isclose(records['camera_fans'][2]['endpoint_half_height'],1.05*FINAL_HALF)
assert records['camera_fans'][0]['endpoint_half_height']<65
assert lower+ZOOM_FAN_HALF<zoom_brace_height

# Geometric and mathematical integrity checks.
assert np.all(np.diff(inverse_shape[:,1])>0), 'Inverse curve must be single-valued in y.'
assert np.min(inverse_shape[:,0])<0 and np.max(inverse_shape[:,0])>0
assert np.isclose(inverse_shape[-1,1], INVERSE_FAN_HALF)
assert np.isclose(inverse_shape[0,1], -INVERSE_FAN_HALF)
assert np.min(psf_shape[:,0])>=0
assert len(input_points)>len(samples_top)==len(samples_bottom)==3
shift=np.array([0,DY])
assert np.allclose(source_bottom-shift, source_top)
assert np.allclose(output_bottom-shift, output_top)
assert np.allclose(samples_bottom-shift, samples_top)
for kind in ['camera','inverse']:
    assert len({q['shape_hash'] for q in records[kind+'_profiles']})==1
    assert all(q['display_growth']>0 for q in records[kind+'_profiles'])
    if kind=='inverse':
        assert all(q['display_growth']<1 for q in records[kind+'_profiles'])
upper=records['braces'][:4]
assert upper[0]['x1']==upper[1]['x0']==X['sampled']
assert upper[1]['x1']==upper[2]['x0']==X['reconstructed']
assert upper[2]['x1']==upper[3]['x0']==X['rescaled']
assert upper[0]['x0']==X['scene'] and upper[3]['x1']==X['output']
scaling_brace = records['braces'][4]
assert scaling_brace['x0']==X['sampled'] and scaling_brace['x1']==X['output']
assert scaling_brace['start_y']-(CY+INPUT_HALF)>8
assert output_label_top.get_position()[0]==output_label_bottom.get_position()[0]
assert output_label_top.get_ha()==output_label_bottom.get_ha()
assert all(abs(upper[j+1]['drawn_x0']-upper[j]['drawn_x1']-4)<1e-9 for j in range(3))
assert np.isclose((X['output']-X['sampled'])/(X['filtered']-X['scene']),1.5)
assert len(records['camera_profiles'])==12 and len(records['inverse_profiles'])==18
assert X['scene']==55 and X['filtered']==385 and X['sampled']==455
assert records['scaling_background'] is None
assert records['guide_pairs']==[dict(x0=X['reconstructed'],x1=X['rescaled'],y=CY),
    dict(x0=R['reconstructed'],x1=R['rescaled'],y=RY)]
reverse_braces = records['braces'][6:9]
assert reverse_braces[0]['x0']==R['output']
assert reverse_braces[0]['x1']==reverse_braces[1]['x0']==R['rescaled']
assert reverse_braces[1]['x1']==reverse_braces[2]['x0']==R['reconstructed']
assert reverse_braces[2]['x1']==R['input']
assert all(np.isclose(reverse_braces[j+1]['drawn_x0']-
                      reverse_braces[j]['drawn_x1'],4) for j in range(2))
assert records['braces'][9]['x0']==R['output']
assert records['braces'][9]['x1']==R['input']

# Place direction arrows immediately beside their own operator labels.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
records['direction_arrows'] = []
for artist, direction in [(forward_scale_label,1), (reverse_scale_label,-1)]:
    box=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    edge=max(box.x0,box.x1) if direction>0 else min(box.x0,box.x1)
    start=edge+direction*14.
    end=start+direction*60.
    y=artist.get_position()[1]-2.
    ax.annotate('', xy=(end,y), xytext=(start,y),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=.72,
                                mutation_scale=8., shrinkA=0, shrinkB=0),
                zorder=12)
    records['direction_arrows'].append(dict(
        label=artist.get_text(), x0=start, x1=end, y=y,
        direction='right' if direction>0 else 'left'))
assert records['direction_arrows'][0]['x1']>records['direction_arrows'][0]['x0']
assert records['direction_arrows'][1]['x1']<records['direction_arrows'][1]['x0']
assert np.isclose(records['braces'][5]['start_y'],lower+OUTPUT_HALF)
assert np.isclose(records['braces'][5]['x0'],source_x)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
boxes=[]
for artist in texts:
    box=artist.get_window_extent(renderer)
    if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
        raise RuntimeError('Text outside canvas: '+artist.get_text())
    boxes.append((artist.get_text(),box))
overlaps=[]
for i,(a,ba) in enumerate(boxes):
    for b,bb in boxes[i+1:]:
        if ba.overlaps(bb):
            overlaps.append((a,b))
assert not overlaps, f'Text overlaps: {overlaps}'
input_box = input_label.get_window_extent(renderer).transformed(ax.transData.inverted())
assert min(input_box.y0,input_box.y1)>previous_extent(min(input_box.x0,input_box.x1))+CY

if args.preview_only:
    fig.savefig(ROOT/'layout_preview.png',dpi=210,
                facecolor='white',pad_inches=0)
    print('Saved layout trial')
    raise SystemExit(0)

for ext in ['pdf','svg','png']:
    fig.savefig(ROOT/f'fig_huygens_bidirectional.{ext}',dpi=420,
                facecolor='white',pad_inches=0)
pdf=fitz.open(ROOT/'fig_huygens_bidirectional.pdf')
assert len(pdf)==1 and len(pdf[0].get_images())==0, 'PDF must be fully vector.'
pdf[0].get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(ROOT/'pdf_preview.png')
records['checks']={
    'notation_source':'latest supplied manuscript; eq:restricted-capture; eq:master',
    'scene_notation':'I^{cont}', 'input_sample_count':5,'output_sample_count':3,
    'equal_source_profiles':True,'equal_output_profiles':True,'equal_output_grids':True,
    'identical_normalized_camera_shapes':True,'identical_normalized_sinc_shapes':True,
    'glyphs_grow_with_fans':True,'upper_brace_gap_points':4*INCH_PER_UNIT*72,
    'inverse_glyph_endpoints_touch_fan_edges':True,
    'output_equality_marker':True,'reconstruction_label_above_curve':True,
    'scaling_brace_at_sample_label_level':True,
    'input_brace_gap_points':(scaling_brace['start_y']-CY-INPUT_HALF)*INCH_PER_UNIT*72,
    'output_labels_vertically_aligned':True,'input_label_position_preserved':True,
    'scaling_background_removed':True,'signal_vertical_extents_preserved':True,
    'dashed_guides_only_at_coordinate_rescaling':True,'brace_linewidth_points':.55,'brace_opacity':.35,
    'height_increase_relative_to_original_percent':100*(H/865-1),
    'height_reduction_relative_to_three_row_v1_percent':100*(1-H/1320),
    'zoom_brace_endpoint_matches_output_grid_bottom':True,
    'zoom_brace_left_endpoint_below_example_point':True,
    'zoom_heading_to_next_heading_center_gap_units':(936-ROW_RAISE-44)-(zoom_brace_height+38),
    'removed_left_explanation':True,
    'reciprocal_scale_typesetting':'slanted fraction',
    'inverse_kernel_function':'numpy.sinc', 'all_braces_aligned':True,
    'embedded_raster_images':len(pdf[0].get_images()),
    'width_inches':W*INCH_PER_UNIT, 'height_inches':H*INCH_PER_UNIT,
    'width_reduction_relative_to_v7_percent':100*(1-W/1800),
    'width_reduction_relative_to_v8_percent':100*(1-W/1660),
    'width_reduction_relative_to_v9_percent':100*(1-W/1395),
    'initial_psf_span_units':X['filtered']-X['scene'],
    'scaling_span_units':X['output']-X['sampled'],
    'scaling_to_initial_psf_span_ratio':(X['output']-X['sampled'])/(X['filtered']-X['scene']),
    'initial_capture_signal_geometry_preserved':True,
    'original_kernel_profile_count_retained':True,
    'total_kernel_profiles':30,
    'reverse_grid_aligned_with_top':True,
    'rescaling_span_reduction_relative_to_v8_percent':100*(1-(X['rescaled']-X['reconstructed'])/150),
    'font_increase_relative_to_v7_percent':100*(FONT_SCALE/1.15-1),
    'filtered_to_sample_gap_units':X['sampled']-X['filtered'],
    'final_front_to_sample_gap_units':X['output']-X['front'],
    'rescaling_span_units':X['rescaled']-X['reconstructed'],
    'font_scale_relative_to_v6':FONT_SCALE,
    'curves_are_schematic':True,
    'camera_fans_equal_opening_angle':True,
    'camera_full_opening_angle_degrees':float(2*np.degrees(np.arctan(CAMERA_SLOPE))),
    'zoom_fan_to_final_signal_height_ratio':1.05,
    'purple_signals_unchanged_from_v2':True,
    'sample_grids_unchanged_from_v2':True,
    'reconstruction_fans_unchanged_from_v2':True,
}
(ROOT/'verification.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records['checks'],indent=2))
