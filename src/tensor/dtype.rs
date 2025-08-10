use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int64 = 4,
    Bool = 5,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 | DType::BFloat16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Bool => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DType::Float32 => "Float32",
            DType::Float16 => "Float16",
            DType::BFloat16 => "BFloat16",
            DType::Int32 => "Int32",
            DType::Int64 => "Int64",
            DType::Bool => "Bool",
        };
        write!(f, "{}", name)
    }
}

pub trait TypeToDType {
    const DTYPE: DType;
}

impl TypeToDType for f32 {
    const DTYPE: DType = DType::Float32;
}

impl TypeToDType for i32 {
    const DTYPE: DType = DType::Int32;
}

impl TypeToDType for i64 {
    const DTYPE: DType = DType::Int64;
}

impl TypeToDType for u8 {
    const DTYPE: DType = DType::Bool;
}

impl TypeToDType for u16 {
    const DTYPE: DType = DType::Float16;
}

pub fn check_dtype_match<T: TypeToDType>(dtype: DType) -> Result<(), String> {
    if T::DTYPE == dtype {
        Ok(())
    } else {
        Err(format!("Type mismatch: expected {:?}, got {:?}", T::DTYPE, dtype))
    }
}

pub type Array1d<T> = Vec<T>;
pub type Array2d<T> = Vec<Vec<T>>;
pub type Array3d<T> = Vec<Vec<Vec<T>>>;

pub fn flatten_2d<T: Clone>(arr: &Array2d<T>) -> Array1d<T> {
    arr.iter().flat_map(|row| row.iter().cloned()).collect()
}

pub fn flatten_3d<T: Clone>(arr: &Array3d<T>) -> Array1d<T> {
    arr.iter()
        .flat_map(|mat| mat.iter())
        .flat_map(|row| row.iter().cloned())
        .collect()
}

pub const MAX_TENSOR_DIM: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct Dim2D {
    pub h: i64,
    pub w: i64,
}

impl Dim2D {
    pub fn new(n: i64) -> Self {
        Self { h: n, w: n }
    }

    pub fn new_hw(h: i64, w: i64) -> Self {
        Self { h, w }
    }
}
