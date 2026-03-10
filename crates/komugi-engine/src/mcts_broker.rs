use std::time::Instant;

use komugi_core::{Color, Move, Position};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BrokerSearchId(u64);

impl BrokerSearchId {
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SearchGameIdentity {
    pub move_number: u32,
    pub side_to_move: Color,
}

impl SearchGameIdentity {
    pub(crate) fn from_position(position: &Position) -> Self {
        Self {
            move_number: position.move_number,
            side_to_move: position.turn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TreeOwnership {
    pub root_node_idx: usize,
}

#[derive(Debug)]
pub(crate) struct PendingLeafTask {
    pub node_idx: usize,
    pub position: Position,
    pub moves: Vec<Move>,
    pub path: Vec<usize>,
    pub is_root: bool,
    pub queued_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchTaskPhase {
    Submit,
    Pause,
    EnqueueLeaves,
    Resume,
    CancelDrain,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaskTransitionError {
    InvalidTransition {
        from: SearchTaskPhase,
        to: SearchTaskPhase,
    },
    InFlightLeavesRemaining {
        in_flight: usize,
    },
}

#[derive(Debug)]
pub(crate) struct BrokerSearchTask {
    search_id: BrokerSearchId,
    game: SearchGameIdentity,
    tree: TreeOwnership,
    phase: SearchTaskPhase,
    pending_leaves: Vec<PendingLeafTask>,
    in_flight_leaves: usize,
}

impl BrokerSearchTask {
    pub(crate) fn submit(
        search_id: BrokerSearchId,
        game: SearchGameIdentity,
        tree: TreeOwnership,
    ) -> Self {
        Self {
            search_id,
            game,
            tree,
            phase: SearchTaskPhase::Submit,
            pending_leaves: Vec::new(),
            in_flight_leaves: 0,
        }
    }

    pub(crate) fn search_id(&self) -> BrokerSearchId {
        self.search_id
    }

    pub(crate) fn game(&self) -> SearchGameIdentity {
        self.game
    }

    pub(crate) fn tree(&self) -> TreeOwnership {
        self.tree
    }

    #[cfg(test)]
    pub(crate) fn phase(&self) -> SearchTaskPhase {
        self.phase
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.pending_leaves.len()
    }

    pub(crate) fn pending_leaves(&self) -> &[PendingLeafTask] {
        &self.pending_leaves
    }

    pub(crate) fn pause(&mut self) -> Result<(), TaskTransitionError> {
        self.transition(SearchTaskPhase::Pause, &[SearchTaskPhase::Submit])
    }

    pub(crate) fn enqueue_leaf(
        &mut self,
        leaf: PendingLeafTask,
    ) -> Result<(), TaskTransitionError> {
        self.transition(
            SearchTaskPhase::EnqueueLeaves,
            &[SearchTaskPhase::Pause, SearchTaskPhase::EnqueueLeaves],
        )?;
        self.pending_leaves.push(leaf);
        self.in_flight_leaves = self.in_flight_leaves.saturating_add(1);
        Ok(())
    }

    pub(crate) fn resume(&mut self) -> Result<Vec<PendingLeafTask>, TaskTransitionError> {
        self.transition(
            SearchTaskPhase::Resume,
            &[SearchTaskPhase::Pause, SearchTaskPhase::EnqueueLeaves],
        )?;
        Ok(std::mem::take(&mut self.pending_leaves))
    }

    pub(crate) fn complete_resumed_leaf(&mut self) -> Result<(), TaskTransitionError> {
        self.transition(SearchTaskPhase::Resume, &[SearchTaskPhase::Resume])?;
        self.in_flight_leaves = self.in_flight_leaves.saturating_sub(1);
        Ok(())
    }

    pub(crate) fn cancel_drain(&mut self) -> Result<Vec<PendingLeafTask>, TaskTransitionError> {
        self.transition(
            SearchTaskPhase::CancelDrain,
            &[
                SearchTaskPhase::Submit,
                SearchTaskPhase::Pause,
                SearchTaskPhase::EnqueueLeaves,
                SearchTaskPhase::Resume,
                SearchTaskPhase::CancelDrain,
            ],
        )?;
        let drained = std::mem::take(&mut self.pending_leaves);
        self.in_flight_leaves = self.in_flight_leaves.saturating_sub(drained.len());
        Ok(drained)
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), TaskTransitionError> {
        self.transition(SearchTaskPhase::Shutdown, &[SearchTaskPhase::CancelDrain])?;
        if self.in_flight_leaves != 0 {
            return Err(TaskTransitionError::InFlightLeavesRemaining {
                in_flight: self.in_flight_leaves,
            });
        }
        Ok(())
    }

    fn transition(
        &mut self,
        next: SearchTaskPhase,
        allowed: &[SearchTaskPhase],
    ) -> Result<(), TaskTransitionError> {
        if allowed.contains(&self.phase) {
            self.phase = next;
            Ok(())
        } else {
            Err(TaskTransitionError::InvalidTransition {
                from: self.phase,
                to: next,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_leaf(node_idx: usize) -> PendingLeafTask {
        PendingLeafTask {
            node_idx,
            position: Position::new(komugi_core::SetupMode::Beginner),
            moves: Vec::new(),
            path: vec![0, node_idx],
            is_root: node_idx == 0,
            queued_at: Instant::now(),
        }
    }

    #[test]
    fn broker_task_happy_path_transitions() {
        let position = Position::new(komugi_core::SetupMode::Beginner);
        let mut task = BrokerSearchTask::submit(
            BrokerSearchId::new(1),
            SearchGameIdentity::from_position(&position),
            TreeOwnership { root_node_idx: 0 },
        );

        assert_eq!(task.phase(), SearchTaskPhase::Submit);
        task.pause().unwrap();
        assert_eq!(task.phase(), SearchTaskPhase::Pause);

        task.enqueue_leaf(test_leaf(3)).unwrap();
        task.enqueue_leaf(test_leaf(4)).unwrap();
        assert_eq!(task.phase(), SearchTaskPhase::EnqueueLeaves);

        let resumed = task.resume().unwrap();
        assert_eq!(resumed.len(), 2);
        assert_eq!(task.phase(), SearchTaskPhase::Resume);

        task.complete_resumed_leaf().unwrap();
        task.complete_resumed_leaf().unwrap();

        let drained = task.cancel_drain().unwrap();
        assert!(drained.is_empty());
        assert_eq!(task.phase(), SearchTaskPhase::CancelDrain);

        task.shutdown().unwrap();
        assert_eq!(task.phase(), SearchTaskPhase::Shutdown);
    }

    #[test]
    fn cancel_drain_drains_pending_leaves() {
        let position = Position::new(komugi_core::SetupMode::Beginner);
        let mut task = BrokerSearchTask::submit(
            BrokerSearchId::new(2),
            SearchGameIdentity::from_position(&position),
            TreeOwnership { root_node_idx: 0 },
        );

        task.pause().unwrap();
        task.enqueue_leaf(test_leaf(1)).unwrap();
        task.enqueue_leaf(test_leaf(2)).unwrap();

        let drained = task.cancel_drain().unwrap();
        assert_eq!(drained.len(), 2);
        assert_eq!(task.pending_len(), 0);

        task.shutdown().unwrap();
    }

    #[test]
    fn shutdown_rejects_unfinished_in_flight_leaves() {
        let position = Position::new(komugi_core::SetupMode::Beginner);
        let mut task = BrokerSearchTask::submit(
            BrokerSearchId::new(3),
            SearchGameIdentity::from_position(&position),
            TreeOwnership { root_node_idx: 0 },
        );

        task.pause().unwrap();
        task.enqueue_leaf(test_leaf(6)).unwrap();
        let _ = task.resume().unwrap();

        task.cancel_drain().unwrap();
        let err = task.shutdown().unwrap_err();
        assert_eq!(
            err,
            TaskTransitionError::InFlightLeavesRemaining { in_flight: 1 }
        );
    }
}
