<script lang="ts">
  // A "pick exactly one of N" pill control.
  //
  // NOT a tablist. The Geometry Lab's two primary controls — which field is drawn, and
  // which layer it is drawn from — were `role="tablist"` wrapped around plain buttons,
  // so nothing carried `aria-selected` and the only signal of the current field or layer
  // was a background colour. A screen-reader user was told "tab list" and then handed N
  // stateless buttons (red-team D F5; issue #7). ARIA 1.2 requires a tablist's owned
  // elements to be `tab`s AND to control panels; these control no panels. What they
  // really are is a radio group: one setting, N mutually exclusive values.
  //
  // Follows the ARIA APG radio-group pattern: the group is a single tab stop, the
  // checked option carries it, and the arrow keys move the selection (selection follows
  // focus, which is the pattern's default and matches what a mouse click here does).
  interface Option {
    value: string;
    label: string;
    disabled?: boolean;
    title?: string;
  }
  interface Props {
    /** Accessible name for the group — the same word the visible label shows. */
    label: string;
    options: Option[];
    value: string;
    onSelect: (value: string) => void;
    testid?: string;
  }
  let { label, options, value, onSelect, testid }: Props = $props();

  let group: HTMLDivElement | null = $state(null);

  const enabled = $derived(options.filter((o) => !o.disabled));

  // Roving tabindex: the checked option is the tab stop. When the checked option is
  // disabled (the Geometry Lab disables `full` while the force field is showing) the
  // first enabled one takes it, so the control never becomes keyboard-unreachable.
  const stop = $derived(
    options.find((o) => o.value === value && !o.disabled)?.value ?? enabled[0]?.value,
  );

  function focusValue(v: string): void {
    const i = options.findIndex((o) => o.value === v);
    group?.querySelectorAll("button")[i]?.focus();
  }

  function onKeyDown(e: KeyboardEvent): void {
    const OWNED = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!OWNED.includes(e.key) || enabled.length === 0) return;
    e.preventDefault();

    const at = enabled.findIndex((o) => o.value === value);
    let next: number;
    if (e.key === "Home") next = 0;
    else if (e.key === "End") next = enabled.length - 1;
    else {
      const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1;
      // `at < 0` when the current value is disabled; start from the first option so
      // the first arrow press still lands somewhere sensible.
      next = ((at < 0 ? 0 : at) + step + enabled.length) % enabled.length;
    }
    const target = enabled[next].value;
    onSelect(target);
    focusValue(target);
  }
</script>

<!-- The group itself must NOT be focusable: the roving tabindex lives on the radios, which
     is what makes the whole control one tab stop. The linter only knows that `radiogroup`
     is an interactive role.  -->
<!-- svelte-ignore a11y_interactive_supports_focus -->
<div
  class="seg"
  role="radiogroup"
  aria-label={label}
  data-testid={testid}
  bind:this={group}
  onkeydown={onKeyDown}
>
  {#each options as o (o.value)}
    <button
      type="button"
      role="radio"
      aria-checked={o.value === value}
      class:active={o.value === value}
      disabled={o.disabled}
      title={o.title}
      tabindex={o.value === stop ? 0 : -1}
      onclick={() => onSelect(o.value)}
    >{o.label}</button>
  {/each}
</div>

<style>
  .seg {
    display: inline-flex;
    gap: 0.2rem;
    padding: 0.2rem;
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
  }
  .seg button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.26rem 0.7rem;
    font-size: 0.76rem;
    font-weight: 500;
  }
  .seg button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
  }
  .seg button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
</style>
