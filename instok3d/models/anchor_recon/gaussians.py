class Gaussians:
    """Batched 3D Gaussians.

    xyz (B, N, 3), scale (B, N, 3), rotation (B, N, 4), opacity (B, N),
    feature (B, N, 3 * (sh_degree + 1) ** 2).
    """

    def __init__(self):
        self.xyz = None
        self.scale = None
        self.rotation = None
        self.opacity = None
        self.feature = None

    def _get_tensor_keys(self):
        return ['xyz', 'scale', 'rotation', 'opacity', 'feature']

    def detach(self):
        g = self.copy()
        for key in self._get_tensor_keys():
            value = getattr(self, key)
            if value is None:
                continue
            setattr(g, key, value.detach())
        return g

    def copy(self):
        g = Gaussians()
        for key in self._get_tensor_keys():
            value = getattr(self, key)
            if value is None:
                continue
            setattr(g, key, value.clone())
        return g
