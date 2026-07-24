<script lang="ts">
  import { nParticles, nSteps } from "../lib/stores";
</script>

<div class="control">
  <span class="label">Particles <b data-testid="particles-value">{$nParticles}</b></span>
  <input
    type="range" min="50" max="2000" step="50" bind:value={$nParticles}
    data-testid="particles-input"
  />
  <span class="label">Sequence length <b data-testid="seqlen-value">{$nSteps}</b> tokens</span>
  <input
    type="range" min="2" max="24" step="1" bind:value={$nSteps}
    data-testid="seqlen-input"
  />
  <span class="hint">Particle swarm for the Sankey view: how many particles, and how many steps before each sequence stops.</span>
</div>

<style>
  .control input[type="range"] { width: 100%; }
  .hint { font-size: 0.72rem; color: var(--text-dim); line-height: 1.35; }
</style>
