// Per-view explorer stores (specs/002 §3b): defaults per the frozen contract, plus
// the geoWeightsToken ↔ sessionStorage round-trip (jsdom provides sessionStorage).
// Modules are re-imported fresh per test so init-from-storage is actually exercised.
import { get } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const KEY = "llm-geometry:geo-weights-token";

async function freshStores() {
  return import("../../src/lib/explorerStores");
}

describe("explorerStores", () => {
  beforeEach(() => {
    vi.resetModules();
    sessionStorage.clear();
  });

  it("has the contract defaults", async () => {
    const s = await freshStores();
    expect(get(s.archModelId)).toBe("HuggingFaceTB/SmolLM2-135M-Instruct");
    expect(get(s.archSystemPrompt)).toBe("");
    expect(get(s.archTemperature)).toBe(0.8);
    expect(get(s.archMaxNewTokens)).toBe(64);
    expect(get(s.archSelectedNode)).toBeNull();
    expect(get(s.archPrompt).length).toBeGreaterThan(0);
    expect(get(s.geoPrompt).length).toBeGreaterThan(0);
    expect(get(s.geoFieldMode)).toBe("next_next");
    expect(get(s.geoLayer)).toBe("full");
    expect(get(s.geoTemperature)).toBe(0);
    expect(get(s.geoTopM)).toBe(1);
    expect(get(s.geoAntisymmetrize)).toBe(false);
    expect(get(s.geoWeightsToken)).toBeNull();
    expect(s.GEO_WEIGHTS_TOKEN_KEY).toBe(KEY);
  });

  it("writes geoWeightsToken to sessionStorage on set and clears it on null", async () => {
    const s = await freshStores();
    s.geoWeightsToken.set("abc123");
    expect(sessionStorage.getItem(KEY)).toBe("abc123");
    s.geoWeightsToken.set("def456");
    expect(sessionStorage.getItem(KEY)).toBe("def456");
    s.geoWeightsToken.set(null);
    expect(sessionStorage.getItem(KEY)).toBeNull();
  });

  it("initializes geoWeightsToken from sessionStorage (refresh keeps weight edits)", async () => {
    sessionStorage.setItem(KEY, "cafe42");
    const s = await freshStores();
    expect(get(s.geoWeightsToken)).toBe("cafe42");
    // ...and the initial subscribe-back doesn't clobber the stored value.
    expect(sessionStorage.getItem(KEY)).toBe("cafe42");
  });
});
