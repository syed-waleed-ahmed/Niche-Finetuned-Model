import type { AccessMode, HealthResponse } from "../api/types";

/**
 * The access mode advertised by `/health`. Older servers omit the field and always
 * required a key, so anything other than `"public"` means `"key"`. `null` means the
 * server has not answered yet (or never did): the UI then lets requests through and
 * relies on the 401 handling.
 */
export function accessMode(health: HealthResponse | null): AccessMode | null {
  if (!health) return null;
  return health.access === "public" ? "public" : "key";
}
