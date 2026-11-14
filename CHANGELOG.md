# Changelog

## Unreleased

- Added regular-grid approximation for eligible univariate continuous bases
  through `Basis`, `BasisBuilder`, and `TermBuilder`, with a fixed grid by
  default and optional tolerance-based refinement.
- `TermBuilder.np` now removes the trends, scales, and diagonalizes its
  P-spline with smoothcon's float64 transforms, so all `k - 2` penalized
  directions are accurate under JAX's default float32.
- The coefficient prior caps a basis's known penalty rank at the directions its
  stored penalty resolves (eigenvalues above `10 * eps` of the largest), so
  raw ill-conditioned penalties in float32 no longer give NaN prior samples.
