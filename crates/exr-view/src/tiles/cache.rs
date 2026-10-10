//! Least-recently-used map under a byte budget:
//! the RAM tile cache, and the slot bookkeeping of the GPU atlas.
//!
//! Every entry carries its byte size. Inserting past the budget evicts the least
//! recently used unpinned entries; pinned entries (the overview levels, the
//! tiles on screen) are never evicted, so pinned data alone can exceed the
//! budget. (The test-only inspectors show that.)

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

struct Entry<V> {
    value: V,
    bytes: usize,
    stamp: u64,
    pinned: bool,
}

/// LRU map with a byte budget and pinning.
pub struct Lru<K, V> {
    map: HashMap<K, Entry<V>>,
    /// `stamp -> key`, oldest first; stamps are unique.
    order: BTreeMap<u64, K>,
    clock: u64,
    used: usize,
    budget: usize,
}

impl<K: Clone + Eq + Hash, V> Lru<K, V> {
    /// An empty cache holding at most `budget` bytes of unpinned entries.
    pub fn new(budget: usize) -> Self {
        Self { map: HashMap::new(), order: BTreeMap::new(), clock: 0, used: 0, budget }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Look `key` up and mark it most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let stamp = self.tick();
        let e = self.map.get_mut(key)?;
        self.order.remove(&e.stamp);
        e.stamp = stamp;
        self.order.insert(stamp, key.clone());
        Some(&e.value)
    }

    /// Look `key` up without touching its recency.
    #[cfg(test)]
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|e| &e.value)
    }

    /// Whether `key` is present.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Insert (or replace) `key` as most recently used, then evict down to the
    /// budget. Returns the evicted entries, oldest first, so a caller that owns
    /// resources per entry (atlas slots) can recycle them.
    pub fn insert(&mut self, key: K, value: V, bytes: usize) -> Vec<(K, V)> {
        let stamp = self.tick();
        let pinned = match self.map.remove(&key) {
            Some(old) => {
                self.order.remove(&old.stamp);
                self.used -= old.bytes;
                old.pinned
            }
            None => false,
        };
        self.used += bytes;
        self.order.insert(stamp, key.clone());
        self.map.insert(key.clone(), Entry { value, bytes, stamp, pinned });
        self.evict(Some(&key))
    }

    /// Evict least recently used unpinned entries until the budget holds,
    /// never evicting `keep`.
    fn evict(&mut self, keep: Option<&K>) -> Vec<(K, V)> {
        let mut out = Vec::new();
        if self.used <= self.budget {
            return out;
        }
        let victims: Vec<(u64, K)> = self
            .order
            .iter()
            .filter(|(_, k)| Some(*k) != keep && self.map.get(*k).is_some_and(|e| !e.pinned))
            .map(|(s, k)| (*s, k.clone()))
            .collect();
        for (stamp, key) in victims {
            if self.used <= self.budget {
                break;
            }
            self.order.remove(&stamp);
            if let Some(e) = self.map.remove(&key) {
                self.used -= e.bytes;
                out.push((key, e.value));
            }
        }
        out
    }

    /// Remove and return the least recently used unpinned entry — for a caller
    /// that manages a fixed pool itself (the atlas recycles its slots this way).
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        let (stamp, key) = self
            .order
            .iter()
            .find(|(_, k)| self.map.get(*k).is_some_and(|e| !e.pinned))
            .map(|(s, k)| (*s, k.clone()))?;
        self.order.remove(&stamp);
        let e = self.map.remove(&key)?;
        self.used -= e.bytes;
        Some((key, e.value))
    }

    /// Remove `key`.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let e = self.map.remove(key)?;
        self.order.remove(&e.stamp);
        self.used -= e.bytes;
        Some(e.value)
    }

    /// Pin or unpin `key` (a no-op for an absent key). Unpinning can put the
    /// cache over budget; the next insert (or [`Self::trim`]) evicts.
    pub fn set_pinned(&mut self, key: &K, pinned: bool) {
        if let Some(e) = self.map.get_mut(key) {
            e.pinned = pinned;
        }
    }

    /// Unpin every entry.
    pub fn unpin_all(&mut self) {
        for e in self.map.values_mut() {
            e.pinned = false;
        }
    }

    /// Evict down to the budget now; returns the evicted entries.
    #[cfg(test)]
    pub fn trim(&mut self) -> Vec<(K, V)> {
        self.evict(None)
    }

    /// Change the budget and evict down to it.
    #[cfg(test)]
    pub fn set_budget(&mut self, budget: usize) -> Vec<(K, V)> {
        self.budget = budget;
        self.evict(None)
    }

    /// Bytes held.
    pub fn used(&self) -> usize {
        self.used
    }

    /// The byte budget.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Whether pinned entries alone keep the cache above its budget.
    #[cfg(test)]
    pub fn over_budget(&self) -> bool {
        self.used > self.budget
    }

    /// Number of entries.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Drop every entry whose key fails `keep`; returns them.
    pub fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) -> Vec<(K, V)> {
        let drop: Vec<K> = self.map.keys().filter(|k| !keep(k)).cloned().collect();
        drop.into_iter().filter_map(|k| self.remove(&k).map(|v| (k, v))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys<V>(v: &[(u32, V)]) -> Vec<u32> {
        v.iter().map(|(k, _)| *k).collect()
    }

    #[test]
    fn evicts_least_recently_used_down_to_the_budget() {
        let mut c = Lru::new(30);
        assert!(c.insert(1, "a", 10).is_empty());
        assert!(c.insert(2, "b", 10).is_empty());
        assert!(c.insert(3, "c", 10).is_empty());
        assert_eq!(c.get(&1), Some(&"a")); // 1 is now the newest
        assert_eq!(keys(&c.insert(4, "d", 10)), [2]);
        assert_eq!(keys(&c.insert(5, "e", 20)), [3, 1]);
        assert_eq!(c.used(), 30);
        assert_eq!(c.len(), 2);
        assert!(c.contains(&4) && c.contains(&5));
    }

    #[test]
    fn peek_does_not_refresh() {
        let mut c = Lru::new(20);
        c.insert(1, (), 10);
        c.insert(2, (), 10);
        assert!(c.peek(&1).is_some());
        assert_eq!(keys(&c.insert(3, (), 10)), [1]);
    }

    #[test]
    fn pinned_entries_survive_and_are_reported_over_budget() {
        let mut c = Lru::new(20);
        c.insert(1, (), 10);
        c.set_pinned(&1, true);
        c.insert(2, (), 10);
        assert_eq!(keys(&c.insert(3, (), 10)), [2], "the pinned oldest entry stays");
        c.set_pinned(&3, true);
        assert!(c.insert(4, (), 10).is_empty(), "never evicts the entry just inserted");
        assert!(c.over_budget());
        c.unpin_all();
        assert_eq!(keys(&c.trim()), [1]);
        assert!(!c.over_budget());
    }

    #[test]
    fn replacing_an_entry_keeps_its_pin_and_updates_bytes() {
        let mut c = Lru::new(100);
        c.insert(1, "a", 10);
        c.set_pinned(&1, true);
        c.insert(1, "b", 40);
        assert_eq!(c.used(), 40);
        assert_eq!(c.peek(&1), Some(&"b"));
        c.set_budget(10);
        assert_eq!(c.len(), 1, "still pinned after replacement");
        assert_eq!(c.remove(&1), Some("b"));
        assert_eq!(c.used(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn pop_lru_skips_pinned_entries() {
        let mut c = Lru::new(usize::MAX);
        c.insert(1, 'a', 1);
        c.insert(2, 'b', 1);
        c.insert(3, 'c', 1);
        c.set_pinned(&1, true);
        c.get(&2);
        assert_eq!(c.pop_lru(), Some((3, 'c')));
        assert_eq!(c.pop_lru(), Some((2, 'b')));
        assert_eq!(c.pop_lru(), None, "only the pinned entry is left");
        assert_eq!(c.used(), 1);
    }

    #[test]
    fn retain_and_budget_shrink() {
        let mut c = Lru::new(100);
        for k in 0..10u32 {
            c.insert(k, k, 10);
        }
        let dropped = c.retain(|k| k % 2 == 0);
        assert_eq!(dropped.len(), 5);
        assert_eq!(c.used(), 50);
        assert_eq!(keys(&c.set_budget(20)), [0, 2, 4]);
        assert_eq!(c.budget(), 20);
    }
}
