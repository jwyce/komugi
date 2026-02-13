use core::mem::size_of;

use komugi_core::Move;

const MIN_ENTRIES: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: u64,
    pub depth: u8,
    pub score: i32,
    pub bound: Bound,
    pub best_move: Option<Move>,
}

#[derive(Debug, Clone)]
pub struct TranspositionTable {
    entries: Vec<Option<Entry>>,
}

impl TranspositionTable {
    pub fn with_size_bytes(bytes: usize) -> Self {
        let entry_size = size_of::<Option<Entry>>().max(1);
        let count = (bytes / entry_size).max(MIN_ENTRIES);
        Self {
            entries: vec![None; count],
        }
    }

    pub fn with_size_mb(mb: usize) -> Self {
        Self::with_size_bytes(mb.saturating_mul(1024 * 1024))
    }

    pub fn clear(&mut self) {
        self.entries.fill(None);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn probe(&self, key: u64) -> Option<&Entry> {
        let idx = self.index(key);
        self.entries[idx].as_ref().filter(|entry| entry.key == key)
    }

    pub fn store(&mut self, entry: Entry) {
        let idx = self.index(entry.key);
        match &self.entries[idx] {
            Some(existing) if existing.key == entry.key && existing.depth > entry.depth => {}
            _ => self.entries[idx] = Some(entry),
        }
    }

    fn index(&self, key: u64) -> usize {
        key as usize % self.entries.len()
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::with_size_mb(16)
    }
}
