use std::io;

/// Quantization constants for NNUE evaluation
pub const QA: i32 = 255; // Accumulator scale
pub const QB: i32 = 64; // Output scale
pub const SCALE: i32 = 400; // Centipawn scale

/// Magic bytes for NNUE file format
const MAGIC: &[u8; 4] = b"GNUE";

/// NNUE network parameters
#[derive(Debug, Clone)]
pub struct NnueParams {
    pub total_features: u32,
    pub hidden1_size: u32,    // 256
    pub hidden2_size: u32,    // 32
    pub ft_bias: Vec<i16>,    // [256]
    pub ft_weights: Vec<i16>, // [TOTAL_FEATURES × 256]
    pub l1_bias: Vec<i32>,    // [32]
    pub l1_weights: Vec<i8>,  // [512 × 32]
    pub l2_bias: i32,         // scalar
    pub l2_weights: Vec<i8>,  // [32]
}

impl NnueParams {
    /// Parse NNUE parameters from binary data
    pub fn from_bytes(data: &[u8]) -> Result<Self, NnueError> {
        if data.len() < 16 {
            return Err(NnueError::InvalidHeader("data too short".into()));
        }

        // Parse header
        let magic = &data[0..4];
        if magic != MAGIC {
            return Err(NnueError::InvalidMagic);
        }

        let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let total_features = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let hidden1_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

        if hidden1_size != 256 {
            return Err(NnueError::InvalidHeader(format!(
                "hidden1_size must be 256, got {}",
                hidden1_size
            )));
        }

        if data.len() < 20 {
            return Err(NnueError::InvalidHeader(
                "data too short for hidden2_size".into(),
            ));
        }

        let hidden2_size = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);

        if hidden2_size != 32 {
            return Err(NnueError::InvalidHeader(format!(
                "hidden2_size must be 32, got {}",
                hidden2_size
            )));
        }

        let mut offset = 20;

        // Feature Transformer (L0)
        let ft_bias_size = 256;
        let ft_bias = read_i16_array(data, &mut offset, ft_bias_size)?;

        let ft_weights_size = (total_features as usize) * 256;
        let ft_weights = read_i16_array(data, &mut offset, ft_weights_size)?;

        // Hidden Layer 1 (L1)
        let l1_bias_size = 32;
        let l1_bias = read_i32_array(data, &mut offset, l1_bias_size)?;

        let l1_weights_size = 512 * 32;
        let l1_weights = read_i8_array(data, &mut offset, l1_weights_size)?;

        // Output Layer (L2)
        let l2_bias = read_i32_scalar(data, &mut offset)?;
        let l2_weights = read_i8_array(data, &mut offset, 32)?;

        Ok(NnueParams {
            total_features,
            hidden1_size,
            hidden2_size,
            ft_bias,
            ft_weights,
            l1_bias,
            l1_weights,
            l2_bias,
            l2_weights,
        })
    }

    /// Serialize NNUE parameters to binary format
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&self.total_features.to_le_bytes());
        buf.extend_from_slice(&self.hidden1_size.to_le_bytes());
        buf.extend_from_slice(&self.hidden2_size.to_le_bytes());

        // Feature Transformer (L0)
        for &val in &self.ft_bias {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        for &val in &self.ft_weights {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        // Hidden Layer 1 (L1)
        for &val in &self.l1_bias {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        for &val in &self.l1_weights {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        // Output Layer (L2)
        buf.extend_from_slice(&self.l2_bias.to_le_bytes());
        for &val in &self.l2_weights {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        buf
    }
}

/// Error types for NNUE format operations
#[derive(Debug)]
pub enum NnueError {
    InvalidMagic,
    InvalidHeader(String),
    InsufficientData(String),
    IoError(io::Error),
}

impl std::fmt::Display for NnueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NnueError::InvalidMagic => write!(f, "Invalid magic bytes (expected GNUE)"),
            NnueError::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            NnueError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            NnueError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for NnueError {}

impl From<io::Error> for NnueError {
    fn from(e: io::Error) -> Self {
        NnueError::IoError(e)
    }
}

// Helper functions for reading binary data with little-endian byte order

fn read_i16_array(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, NnueError> {
    let bytes_needed = count * 2;
    if *offset + bytes_needed > data.len() {
        return Err(NnueError::InsufficientData(format!(
            "need {} bytes, have {}",
            bytes_needed,
            data.len() - *offset
        )));
    }

    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let bytes = [data[*offset], data[*offset + 1]];
        result.push(i16::from_le_bytes(bytes));
        *offset += 2;
    }
    Ok(result)
}

fn read_i32_array(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i32>, NnueError> {
    let bytes_needed = count * 4;
    if *offset + bytes_needed > data.len() {
        return Err(NnueError::InsufficientData(format!(
            "need {} bytes, have {}",
            bytes_needed,
            data.len() - *offset
        )));
    }

    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let bytes = [
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ];
        result.push(i32::from_le_bytes(bytes));
        *offset += 4;
    }
    Ok(result)
}

fn read_i8_array(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, NnueError> {
    if *offset + count > data.len() {
        return Err(NnueError::InsufficientData(format!(
            "need {} bytes, have {}",
            count,
            data.len() - *offset
        )));
    }

    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(data[*offset] as i8);
        *offset += 1;
    }
    Ok(result)
}

fn read_i32_scalar(data: &[u8], offset: &mut usize) -> Result<i32, NnueError> {
    if *offset + 4 > data.len() {
        return Err(NnueError::InsufficientData(
            "need 4 bytes for i32 scalar".into(),
        ));
    }

    let bytes = [
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ];
    *offset += 4;
    Ok(i32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_serialization() {
        let original = NnueParams {
            total_features: 768,
            hidden1_size: 256,
            hidden2_size: 32,
            ft_bias: vec![1i16; 256],
            ft_weights: vec![2i16; 768 * 256],
            l1_bias: vec![3i32; 32],
            l1_weights: vec![4i8; 512 * 32],
            l2_bias: 5i32,
            l2_weights: vec![6i8; 32],
        };

        let bytes = original.to_bytes();
        let restored = NnueParams::from_bytes(&bytes).expect("failed to deserialize");

        assert_eq!(original.total_features, restored.total_features);
        assert_eq!(original.hidden1_size, restored.hidden1_size);
        assert_eq!(original.hidden2_size, restored.hidden2_size);
        assert_eq!(original.ft_bias, restored.ft_bias);
        assert_eq!(original.ft_weights, restored.ft_weights);
        assert_eq!(original.l1_bias, restored.l1_bias);
        assert_eq!(original.l1_weights, restored.l1_weights);
        assert_eq!(original.l2_bias, restored.l2_bias);
        assert_eq!(original.l2_weights, restored.l2_weights);
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"XXXX");
        let result = NnueParams::from_bytes(&data);
        assert!(matches!(result, Err(NnueError::InvalidMagic)));
    }

    #[test]
    fn test_invalid_hidden1_size() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(MAGIC);
        data[12..16].copy_from_slice(&128u32.to_le_bytes()); // wrong size
        let result = NnueParams::from_bytes(&data);
        assert!(matches!(result, Err(NnueError::InvalidHeader(_))));
    }

    #[test]
    fn test_invalid_hidden2_size() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(MAGIC);
        data[12..16].copy_from_slice(&256u32.to_le_bytes()); // correct hidden1
        data[16..20].copy_from_slice(&64u32.to_le_bytes()); // wrong hidden2
        let result = NnueParams::from_bytes(&data);
        assert!(matches!(result, Err(NnueError::InvalidHeader(_))));
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![0u8; 10];
        let result = NnueParams::from_bytes(&data);
        assert!(matches!(result, Err(NnueError::InvalidHeader(_))));
    }

    #[test]
    fn test_little_endian_encoding() {
        let original = NnueParams {
            total_features: 512,
            hidden1_size: 256,
            hidden2_size: 32,
            ft_bias: vec![0i16; 256],
            ft_weights: vec![0i16; 512 * 256],
            l1_bias: vec![0i32; 32],
            l1_weights: vec![0i8; 512 * 32],
            l2_bias: 0i32,
            l2_weights: vec![0i8; 32],
        };

        let bytes = original.to_bytes();
        let params = NnueParams::from_bytes(&bytes).expect("failed to parse");
        assert_eq!(params.total_features, 512);
    }

    #[test]
    fn test_negative_values() {
        let original = NnueParams {
            total_features: 100,
            hidden1_size: 256,
            hidden2_size: 32,
            ft_bias: vec![-100i16; 256],
            ft_weights: vec![-50i16; 100 * 256],
            l1_bias: vec![-200i32; 32],
            l1_weights: vec![-10i8; 512 * 32],
            l2_bias: -500i32,
            l2_weights: vec![-5i8; 32],
        };

        let bytes = original.to_bytes();
        let restored = NnueParams::from_bytes(&bytes).expect("failed to deserialize");

        assert_eq!(original.ft_bias, restored.ft_bias);
        assert_eq!(original.ft_weights, restored.ft_weights);
        assert_eq!(original.l1_bias, restored.l1_bias);
        assert_eq!(original.l1_weights, restored.l1_weights);
        assert_eq!(original.l2_bias, restored.l2_bias);
        assert_eq!(original.l2_weights, restored.l2_weights);
    }
}
