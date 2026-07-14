# Vendored DA2 runtime

- Upstream: https://github.com/EnVision-Research/DA-2
- Revision: `d659838585f2bc9967c7e9367af573795271dbbb`
- License: Apache License 2.0 (`LICENSE`)
- Scope: model runtime plus the official inference architecture configuration.

The package initializer is intentionally minimal so VLN inference does not import
DA2 visualization and point-cloud dependencies. `SphereViT` also exposes the
spherical decoder feature pyramid used by the VLN geometry adapter.
