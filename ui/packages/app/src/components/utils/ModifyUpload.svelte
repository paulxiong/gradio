<script lang="ts">
	export let edit: (() => void) | false = false;
	export let clear: () => void = () => {};
	export let theme: string;
	export let static_src: string;
	export let absolute = true;
</script>

<div
	class="modify-upload z-10 top-0 right-0 flex justify-end"
	class:absolute
	{theme}
>
	{#if edit}
		<button
			class="edit bg-opacity-30 hover:bg-opacity-100 transition p-1"
			on:click={edit}
		>
			<img
				class="h-4 filter dark:invert"
				src="{static_src}/static/img/edit.svg"
				alt="Edit"
			/>
		</button>
	{/if}
	<button
		class="clear bg-opacity-30 hover:bg-opacity-100 transition p-1"
		on:click={clear}
	>
		<img
			class="h-4 filter dark:invert"
			src="{static_src}/static/img/clear.svg"
			alt="Clear"
		/>
	</button>
</div>

<style lang="postcss" global>
	.modify-upload[theme="default"] {
		@apply m-1 flex gap-1;
		.edit {
			@apply bg-amber-500 dark:bg-red-600 rounded shadow;
		}
		.clear {
			@apply bg-gray-50 dark:bg-gray-500 rounded shadow;
		}
	}
</style>
