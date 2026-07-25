// Vite client env typing (import.meta.env.*) for the static-mode data layer.
// The project tsconfig pins `types` to vitest/globals, so the vite/client lib is
// pulled in explicitly here (picked up via the tsconfig `include` globs).
/// <reference types="vite/client" />
