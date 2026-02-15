import { Gungi, BEGINNER_POSITION, INTRO_POSITION, INTERMEDIATE_POSITION, ADVANCED_POSITION } from '/Users/jwyce/dev/projects/gungi.js/dist/index.js';
import { writeFileSync } from 'node:fs';

function stripGameOverSuffix(san) {
	return san.replace(/[#=]$/g, '');
}

function snapshot(gungi) {
	const rawMoves = gungi.moves();
	const sans = rawMoves.map((m) => stripGameOverSuffix(typeof m === 'string' ? m : m.san)).sort();

	return {
		fen: gungi.fen(),
		moves: sans,
		move_count: sans.length,
		in_check: gungi.inCheck(),
		is_checkmate: gungi.isCheckmate(),
		is_stalemate: gungi.isStalemate(),
		is_draw: gungi.isDraw(),
		is_game_over: gungi.isGameOver(),
	};
}

function playRandomGame(startFen, maxMoves, sampleEvery) {
	const gungi = new Gungi(startFen);
	const vectors = [];
	let ply = 0;

	vectors.push(snapshot(gungi));

	while (!gungi.isGameOver() && ply < maxMoves) {
		const moves = gungi.moves();
		if (moves.length === 0) break;
		const move = moves[Math.floor(Math.random() * moves.length)];
		gungi.move(move);
		ply++;

		if (ply % sampleEvery === 0 || gungi.isGameOver() || gungi.inCheck()) {
			vectors.push(snapshot(gungi));
		}
	}

	if (gungi.isGameOver()) {
		const last = vectors[vectors.length - 1];
		if (last.fen !== gungi.fen()) {
			vectors.push(snapshot(gungi));
		}
	}

	return vectors;
}

const allVectors = [];
const seed = Date.now();
console.log(`Seed: ${seed}`);

const configs = [
	{ name: 'intro', fen: INTRO_POSITION, games: 10, maxMoves: 300, sampleEvery: 3 },
	{ name: 'beginner', fen: BEGINNER_POSITION, games: 10, maxMoves: 300, sampleEvery: 3 },
	{ name: 'intermediate', fen: INTERMEDIATE_POSITION, games: 3, maxMoves: 200, sampleEvery: 5 },
	{ name: 'advanced', fen: ADVANCED_POSITION, games: 3, maxMoves: 200, sampleEvery: 5 },
];

for (const cfg of configs) {
	console.log(`Playing ${cfg.games} ${cfg.name} games...`);
	for (let i = 0; i < cfg.games; i++) {
		const vectors = playRandomGame(cfg.fen, cfg.maxMoves, cfg.sampleEvery);
		allVectors.push(...vectors);
		process.stdout.write(`.`);
	}
	console.log(` ${allVectors.length} vectors so far`);
}

const seen = new Set();
const deduped = allVectors.filter((v) => {
	if (seen.has(v.fen)) return false;
	seen.add(v.fen);
	return true;
});

console.log(`\nTotal: ${allVectors.length} vectors, ${deduped.length} unique by FEN`);

const outPath = new URL('./cross_engine_vectors.json', import.meta.url).pathname;
writeFileSync(outPath, JSON.stringify(deduped, null, 2));
console.log(`Written to ${outPath}`);
