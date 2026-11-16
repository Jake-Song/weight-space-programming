"""Build the experiment's schematic text backbone with Blender 4.5's bpy.

Run with Blender --background --python scripts/build_model_3d.py, or Python
with the official bpy module installed. No model weights are required.
"""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'sites/weight-space-programming/dist/assets'
CONFIG = json.loads((OUT / 'model-config.json').read_text())['text_config']
AUDIT = json.loads((ROOT / 'results/weight_audit.json').read_text())
CHANGED = {int(t['name'].split('.layers.')[1].split('.')[0]): t['entries']
           for a in AUDIT if a['condition'] == 'both' for t in a['changed']}
assert CONFIG['num_hidden_layers'] == 24
assert set(CHANGED) == {15, 19}

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene['description'] = 'Qwen3.5-0.8B text backbone schematic, 0-based indices. Only MLP down_proj in layers 15 and 19 was edited. Geometry is not tensor magnitude or activation data.'
scene['model_revision'] = '2fc06364715b967f1860aea9cf38778875588b17'

def material(name, color, metallic=0.0, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = .36
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Emission Color'].default_value = (*color, 1)
    bsdf.inputs['Emission Strength'].default_value = emission
    return mat

base = material('Layer chassis / unchanged', (.045, .065, .1), .5)
attention = material('Attention / unchanged', (.09, .15, .22), .45)
mlp = material('MLP / unchanged', (.1, .12, .18), .4)
white = material('Typography', (.55, .66, .78), .05, .4)
dim = material('Data path / schematic', (.11, .2, .27), .5, .2)
purple = material('EDIT / layer 15', (.48, .28, 1), .25, 1.2)
mint = material('EDIT / layer 19', (.22, .86, .61), .25, 1.1)
materials = {15: purple, 19: mint}
chassis_materials = {
    15: material('Layer 15 / violet chassis', (.15, .065, .3), .4, .15),
    19: material('Layer 19 / mint chassis', (.025, .23, .15), .4, .15),
}
roots = []

def cube(name, location, size, mat, parent=None, bevel=.035):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new('Soft machined edges', 'BEVEL')
        mod.width = bevel
        mod.segments = 2
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
    if parent:
        obj.parent = parent
    return obj

def label(name, text, location, size, mat, parent=None, front=True):
    curve = bpy.data.curves.new(name, 'FONT')
    curve.body = text
    curve.size = size
    curve.extrude = .001
    curve.align_x = 'CENTER'
    obj = bpy.data.objects.new(name, curve)
    scene.collection.objects.link(obj)
    obj.location = location
    if front:
        obj.rotation_euler = (math.pi / 2, 0, 0)
    obj.data.materials.append(mat)
    obj.parent = parent
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    obj.select_set(False)
    return obj

for index in range(24):
    root = bpy.data.objects.new(f'Layer_{index:02d}', None)
    scene.collection.objects.link(root)
    root.location.z = 1.35 + index * .49
    root['layerIndex'] = index
    root['edited'] = index in CHANGED
    root['attentionType'] = CONFIG['layer_types'][index]
    root['changedValues'] = CHANGED.get(index, 0)
    roots.append(root)
    color = materials.get(index, dim)
    cube(f'L{index:02d}_chassis', (0, 0, 0), (4.6, 3.1, .12), chassis_materials.get(index, base), root)
    attn = cube(f'L{index:02d}_attention', (-1.1, 0, .12), (1.75, 2.58, .14), attention, root)
    attn['component'] = CONFIG['layer_types'][index]
    m = cube(f'L{index:02d}_mlp', (1.05, 0, .12), (1.85, 2.58, .14), mlp, root)
    m['component'] = 'mlp'
    cube(f'L{index:02d}_front_strip', (0, -1.56, -.01), (4.35, .035, .045), color, root, .01)
    label(f'L{index:02d}_index', f'{index:02d}', (-2.08, -1.59, -.045), .15, materials.get(index, white), root)
    if index in CHANGED:
        for x in [-2.29, 2.29]:
            cube(f'L{index:02d}_side_highlight', (x, 0, 0), (.035, 3.1, .07), color, root, .008)
        cube(f'L{index:02d}_badge', (2.85, -.9, .02), (.9, 1.1, .18), chassis_materials[index], root)
        label(f'L{index:02d}_badge_number', str(index), (2.85, -1.47, -.07), .3, color, root)
        # The 8 x 8 marks each represent one edited column, not its actual index.
        for row in range(8):
            for col in range(8):
                cube(f'L{index:02d}_down_proj_column_{row*8+col:02d}',
                     (.32 + col*.205, -.96 + row*.275, .24),
                     (.105, .16, .065), color, root, .012)
        for x in [.08, 2.03]:
            cube(f'L{index:02d}_mlp_outline', (x, 0, .235), (.025, 2.6, .025), color, root, .005)
        for y in [-1.29, 1.29]:
            cube(f'L{index:02d}_mlp_outline', (1.05, y, .235), (1.95, .025, .025), color, root, .005)
        label(f'L{index:02d}_edited_label', f'LAYER {index} / MLP EDIT', (0.35, -1.59, -.045), .14, color, root)
    else:
        # Six understated traces indicate modules, not individual parameters.
        for x in [-1.65, -1.1, -.55, .5, 1.05, 1.6]:
            cube(f'L{index:02d}_trace', (x, 0, .201), (.028, 2.1, .012), dim, root, .003)

for name, z, text in [('Embedding', .57, 'TOKEN EMBEDDING'), ('Output', 13.4, 'NORM + LM HEAD')]:
    root = bpy.data.objects.new(name, None)
    scene.collection.objects.link(root)
    root.location.z = z
    root['stage'] = name.lower()
    cube(name+'_housing', (0, 0, 0), (4.85, 3.35, .29), base, root, .09)
    label(name+'_label', text, (0, -1.7, -.05), .21, white, root)
    for x in [-1.7, -1.2, -.7, -.2, .3, .8, 1.3, 1.8]:
        cube(name+'_contacts', (x, -.2, .18), (.24, 2.3, .05), attention, root)

cube('Display plinth', (0, 0, -.05), (6.4, 4.6, .25), base, bevel=.12)
label('Plinth title', 'QWEN 3.5 / 0.8B', (0, -2.32, -.07), .26, white)
for x in [-2.5, 2.5]:
    cube('Schematic residual spine', (x, 1.27, 6.8), (.055, .055, 12.5), dim, bevel=.01)

# Keep the editable layer coordinates intact under a horizontal presentation root.
backbone = bpy.data.objects.new('Horizontal_text_backbone', None)
scene.collection.objects.link(backbone)
for obj in list(scene.objects):
    if obj != backbone and obj.parent is None:
        obj.parent = backbone
backbone.rotation_euler.y = math.pi / 2
backbone.location.x = -6.7
backbone['layout'] = 'horizontal; input at left, output at right'

# Camera and lighting also remain in the editable .blend scene.
bpy.ops.object.camera_add(location=(8, -27, 16))
camera = bpy.context.object
camera.name = 'Presentation camera'
camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 18
scene.camera = camera
for name, pos, energy, size, color in [
    ('Key', (4, -8, 16), 2000, 9, (.75, .85, 1)),
    ('Fill', (-8, -4, 7), 1800, 8, (.55, .62, 1)),
    ('Rim', (5, 7, 14), 2600, 7, (.45, 1, .8)),
]:
    bpy.ops.object.light_add(type='AREA', location=pos)
    light = bpy.context.object
    light.name = name
    light.data.energy, light.data.size, light.data.color = energy, size, color
    light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat('-Z', 'Y').to_euler()

scene.world = bpy.data.worlds.new('Midnight studio')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.04, .055, .085, 1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value = .55
scene.render.engine = 'CYCLES'
scene.cycles.samples = 24
scene.cycles.use_denoising = True
scene.render.resolution_x = 1500
scene.render.resolution_y = 850
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True
scene.render.filepath = str(OUT / 'qwen-layers.png')

for area in bpy.context.screen.areas if bpy.context.screen else []:
    if area.type == 'VIEW_3D':
        area.spaces.active.region_3d.view_perspective = 'CAMERA'
        area.spaces.active.shading.type = 'MATERIAL'
bpy.ops.object.select_all(action='DESELECT')
for obj in scene.objects:
    if obj.type in {'MESH', 'EMPTY'}:
        obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(OUT / 'qwen-layers.glb'),
    export_format='GLB', use_selection=True, export_extras=True,
    export_cameras=False, export_lights=False, export_animations=False)
bpy.ops.object.select_all(action='DESELECT')
roots[15].select_set(True)
roots[19].select_set(True)
bpy.context.view_layer.objects.active = roots[19]
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / 'qwen-layers.blend'))
metadata = {
    'generator': f'Blender {bpy.app.version_string}',
    'model': 'Qwen/Qwen3.5-0.8B', 'revision': scene['model_revision'],
    'scope': 'Text backbone; schematic geometry, not activation or weight magnitudes',
    'indexing': 'zero-based', 'layout': 'horizontal', 'layers': CONFIG['num_hidden_layers'],
    'hiddenSize': CONFIG['hidden_size'], 'intermediateSize': CONFIG['intermediate_size'],
    'attentionTypes': CONFIG['layer_types'],
    'edits': [{'layer': i, 'columns': 64, 'changedValues': n,
               'tensor': f'model.language_model.layers.{i}.mlp.down_proj.weight',
               'shape': [1024, 3584]} for i, n in CHANGED.items()],
}
(OUT / 'model-metadata.json').write_text(json.dumps(metadata, indent=2))
print('BLEND_AND_GLB_COMPLETE', flush=True)
bpy.ops.render.render(write_still=True)
print('RENDER_COMPLETE', flush=True)
