/// Chess.com-style win probability, accuracy, and garbage time utilities
use std::f64::consts::E;

/// Chess.com winning chances formula
/// Converts centipawn evaluation to winning probability (-1.0 to 1.0)
/// Range: -1.0 (losing) to 1.0 (winning)
pub fn winning_chances(cp: i32) -> f64 {
    let cp_f = cp as f64;
    2.0 / (1.0 + E.powf(-0.004 * cp_f)) - 1.0
}

/// Convert winning chances to win percentage (0-100)
/// Range: 0.0 (losing) to 100.0 (winning)
pub fn win_percent(cp: i32) -> f64 {
    (winning_chances(cp) + 1.0) / 2.0 * 100.0
}

/// Calculate win percentage change between two evaluations
/// Positive value = win% improved, negative = win% worsened
pub fn win_percent_loss(eval_before: i32, eval_after: i32) -> f64 {
    win_percent(eval_before) - win_percent(eval_after)
}

/// Accuracy formula based on centipawn loss (CPL)
/// Range: 0.0 to 100.0
/// K=120 is the standard Chess.com constant
pub fn accuracy_for_cpl(cpl: f64) -> f64 {
    100.0 * (-cpl.abs() / 120.0).exp()
}

/// Check if position is in garbage time (one side has overwhelming advantage)
/// Garbage time threshold: ±700 centipawns
pub fn is_garbage_time(cp: i32) -> bool {
    cp.abs() >= 700
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winning_chances_zero() {
        let wc = winning_chances(0);
        assert!(
            (wc - 0.0).abs() < 0.01,
            "winning_chances(0) should be ~0.0, got {}",
            wc
        );
    }

    #[test]
    fn test_winning_chances_positive() {
        let wc = winning_chances(1000);
        assert!(
            wc > 0.9,
            "winning_chances(1000) should be > 0.9, got {}",
            wc
        );
    }

    #[test]
    fn test_winning_chances_negative() {
        let wc = winning_chances(-1000);
        assert!(
            wc < -0.9,
            "winning_chances(-1000) should be < -0.9, got {}",
            wc
        );
    }

    #[test]
    fn test_win_percent_zero() {
        let wp = win_percent(0);
        assert!(
            (wp - 50.0).abs() < 0.1,
            "win_percent(0) should be ~50.0, got {}",
            wp
        );
    }

    #[test]
    fn test_win_percent_positive() {
        let wp = win_percent(1000);
        assert!(wp > 95.0, "win_percent(1000) should be > 95.0, got {}", wp);
    }

    #[test]
    fn test_win_percent_negative() {
        let wp = win_percent(-1000);
        assert!(wp < 5.0, "win_percent(-1000) should be < 5.0, got {}", wp);
    }

    #[test]
    fn test_win_percent_loss() {
        let loss = win_percent_loss(0, 100);
        assert!(
            loss < 0.0,
            "win_percent_loss(0, 100) should be negative (eval improved)"
        );

        let gain = win_percent_loss(100, 0);
        assert!(
            gain > 0.0,
            "win_percent_loss(100, 0) should be positive (eval worsened)"
        );
    }

    #[test]
    fn test_accuracy_for_cpl_zero() {
        let acc = accuracy_for_cpl(0.0);
        assert!(
            (acc - 100.0).abs() < 0.01,
            "accuracy_for_cpl(0.0) should be 100.0, got {}",
            acc
        );
    }

    #[test]
    fn test_accuracy_for_cpl_positive() {
        let acc = accuracy_for_cpl(50.0);
        assert!(
            acc > 0.0 && acc < 100.0,
            "accuracy_for_cpl(50.0) should be between 0 and 100, got {}",
            acc
        );
    }

    #[test]
    fn test_accuracy_for_cpl_negative() {
        let acc = accuracy_for_cpl(-50.0);
        assert!(
            acc > 0.0 && acc < 100.0,
            "accuracy_for_cpl(-50.0) should be between 0 and 100, got {}",
            acc
        );
    }

    #[test]
    fn test_is_garbage_time_true() {
        assert!(is_garbage_time(700), "is_garbage_time(700) should be true");
        assert!(is_garbage_time(800), "is_garbage_time(800) should be true");
        assert!(
            is_garbage_time(-700),
            "is_garbage_time(-700) should be true"
        );
        assert!(
            is_garbage_time(-800),
            "is_garbage_time(-800) should be true"
        );
    }

    #[test]
    fn test_is_garbage_time_false() {
        assert!(
            !is_garbage_time(500),
            "is_garbage_time(500) should be false"
        );
        assert!(
            !is_garbage_time(699),
            "is_garbage_time(699) should be false"
        );
        assert!(
            !is_garbage_time(-500),
            "is_garbage_time(-500) should be false"
        );
        assert!(
            !is_garbage_time(-699),
            "is_garbage_time(-699) should be false"
        );
    }

    #[test]
    fn test_is_garbage_time_zero() {
        assert!(!is_garbage_time(0), "is_garbage_time(0) should be false");
    }
}
