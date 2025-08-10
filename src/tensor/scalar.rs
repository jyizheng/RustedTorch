use crate::tensor::dtype::DType;

#[derive(Debug, Clone)]
pub enum Scalar {
    Float32(f32),
    Float16(u16),
    BFloat16(u16),
    Int32(i32),
    Int64(i64),
    Bool(bool),
}

impl Scalar {
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::Float32(_) => DType::Float32,
            Scalar::Float16(_) => DType::Float16,
            Scalar::BFloat16(_) => DType::BFloat16,
            Scalar::Int32(_) => DType::Int32,
            Scalar::Int64(_) => DType::Int64,
            Scalar::Bool(_) => DType::Bool,
        }
    }

    pub fn to<T>(&self) -> T
    where
        T: From<f32> + From<i32> + From<i64> + From<u8> + From<u16>,
    {
        match self {
            Scalar::Float32(v) => T::from(*v),
            Scalar::Float16(v) => T::from(*v),
            Scalar::BFloat16(v) => T::from(*v),
            Scalar::Int32(v) => T::from(*v),
            Scalar::Int64(v) => T::from(*v as i32),
            Scalar::Bool(v) => T::from(*v as u8),
        }
    }

    pub fn to_f32(&self) -> f32 {
        match self {
            Scalar::Float32(v) => *v,
            Scalar::Float16(v) => *v as f32,
            Scalar::BFloat16(v) => *v as f32,
            Scalar::Int32(v) => *v as f32,
            Scalar::Int64(v) => *v as f32,
            Scalar::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }

    pub fn to_i64(&self) -> i64 {
        match self {
            Scalar::Float32(v) => *v as i64,
            Scalar::Float16(v) => *v as i64,
            Scalar::BFloat16(v) => *v as i64,
            Scalar::Int32(v) => *v as i64,
            Scalar::Int64(v) => *v,
            Scalar::Bool(v) => if *v { 1 } else { 0 },
        }
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Scalar::Float32(v)
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Scalar::Int32(v)
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::Int64(v)
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}
