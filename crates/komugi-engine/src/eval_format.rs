use komugi_core::Color;

pub const MATE_SCORE: i32 = 30_000;

/// Check if a score represents a forced mate.
/// Mate scores are within 100 points of MATE_SCORE.
pub fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE_SCORE - 100
}

/// Extract mate-in-N from a score.
/// Returns Some(n) where n is positive for mate (white winning) or negative for mated (white losing).
/// Returns None if the score is not a mate score.
pub fn mate_in_n(score: i32) -> Option<i32> {
    if !is_mate_score(score) {
        return None;
    }

    let n = (MATE_SCORE - score.abs() + 1) / 2;
    if score > 0 {
        Some(n)
    } else {
        Some(-n)
    }
}

/// Format a score for display.
/// Mate scores display as "M3", "-M5", etc.
/// Regular scores display as "+1.50", "-0.32", "+0.00", etc.
pub fn format_score(score: i32) -> String {
    if let Some(n) = mate_in_n(score) {
        if n > 0 {
            format!("M{}", n)
        } else {
            format!("-M{}", -n)
        }
    } else if score == 0 {
        "0.00".to_string()
    } else {
        let centipawns = score as f64 / 100.0;
        if centipawns >= 0.0 {
            format!("+{:.2}", centipawns)
        } else {
            format!("{:.2}", centipawns)
        }
    }
}

/// Format a score for display from a specific player's perspective.
/// If turn is Black, the score is negated (flips perspective).
pub fn format_score_for_display(score: i32, turn: Color) -> String {
    let display_score = match turn {
        Color::White => score,
        Color::Black => -score,
    };
    format_score(display_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mate_score() {
        assert!(is_mate_score(29_998)); // mate in 1
        assert!(is_mate_score(-29_998)); // mated in 1
        assert!(is_mate_score(29_900)); // mate in 50
        assert!(!is_mate_score(500)); // regular score
        assert!(!is_mate_score(0)); // zero
        assert!(!is_mate_score(-500)); // regular score
    }

    #[test]
    fn test_mate_in_n() {
        // Mate in 1: MATE_SCORE - 1 = 29_999
        assert_eq!(mate_in_n(29_999), Some(1));
        // Mate in 2: MATE_SCORE - 3 = 29_997
        assert_eq!(mate_in_n(29_997), Some(2));
        // Mated in 1: -(MATE_SCORE - 1) = -29_999
        assert_eq!(mate_in_n(-29_999), Some(-1));
        // Mated in 2: -(MATE_SCORE - 3) = -29_997
        assert_eq!(mate_in_n(-29_997), Some(-2));
        // Regular score
        assert_eq!(mate_in_n(500), None);
        assert_eq!(mate_in_n(0), None);
    }

    #[test]
    fn test_format_score() {
        // Mate scores
        assert_eq!(format_score(29_999), "M1");
        assert_eq!(format_score(29_997), "M2");
        assert_eq!(format_score(-29_999), "-M1");
        assert_eq!(format_score(-29_997), "-M2");

        // Regular scores
        assert_eq!(format_score(150), "+1.50");
        assert_eq!(format_score(-150), "-1.50");
        assert_eq!(format_score(0), "0.00");
        assert_eq!(format_score(32), "+0.32");
        assert_eq!(format_score(-32), "-0.32");
    }

    #[test]
    fn test_format_score_for_display() {
        // White's perspective
        assert_eq!(format_score_for_display(150, Color::White), "+1.50");
        assert_eq!(format_score_for_display(-150, Color::White), "-1.50");

        // Black's perspective (negated)
        assert_eq!(format_score_for_display(150, Color::Black), "-1.50");
        assert_eq!(format_score_for_display(-150, Color::Black), "+1.50");

        // Mate scores
        assert_eq!(format_score_for_display(29_999, Color::White), "M1");
        assert_eq!(format_score_for_display(29_999, Color::Black), "-M1");
    }
}
