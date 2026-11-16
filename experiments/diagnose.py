# Diagnose: confirm the architecture and inspect MLP weight norms without running the model.
print('Tensors:', len(NAMES))
for layer in (15, 19):
    name = f'model.language_model.layers.{layer}.mlp.down_proj.weight'
    w = load_original(name)
    print(name, tuple(w.shape), 'norm:', w.norm().item())
print('Current edits:', edited())
