use flux_rs::attrs::*;
use crate::rvec::RVec;

#[refined_by(start: int, end: int)]
pub struct UsizeRange {
    #[field(usize[start])]
    start: usize,
    #[field(usize[end])]
    end: usize,
}

#[spec(fn(start: usize, end: usize) -> UsizeRange[start, end])]
pub fn spread_usize(start: usize, end: usize) -> UsizeRange {
    UsizeRange {
        start,
        end
    }
}

#[spec(fn(start: i32, end: i32) -> I32Range[start, end])]
pub fn spread_i32(start: i32, end: i32) -> I32Range {
    I32Range {
        start,
        end
    }
}

#[refined_by(start: int, end: int)]
pub struct I32Range {
    #[field(i32[start])]
    start: i32,
    #[field(i32[end])]
    end: i32,
}

// The below are "default" implementations of the associated refinements
// for the `Step` trait, that we put in so that types for which no explicit
// implementation is given can be analyzed without Flux complaining about missing
// implementations. Note that the implementations are "uninterpreted" to make verification
// sound. However, you may get "false positives" if you use these defaults.

defs! {
    fn default_step_step_forward<T>(start: T, count: int) -> T;
    fn default_step_size<T>(lo: T, hi: T) -> int;
}

/// We define the following associated refinements for the `Step` trait, which are then
/// used to specify the API for the `Iterator` implementation for `Range<A>`.
///  - `step_forward` computes the new value after stepping forward from `start` by `count`,
///  - `size` computes the number of steps needed to go from `lo` to `hi
#[assoc(
    fn step_forward(start: Self, count: int) -> Self {
        default_step_step_forward(start, count)
    }
    fn size(lo: Self, hi: Self) -> int {
        default_step_size(lo, hi)
    }
)]
trait Step {}

#[assoc(
    fn step_forward(start: int, count: int) -> int { start + count }
    fn size(lo: int, hi: int) -> int { hi - lo }
)]
impl Step for usize {}

#[assoc(
    fn step_forward(start: int, count: int) -> int { start + count }
    fn size(lo: int, hi: int) -> int { hi - lo }
)]
impl Step for i32 {}

#[assoc(
    fn valid_item(self: UsizeRange, item: int) -> bool { self.start <= item && item < self.end }
    fn size(self: UsizeRange) -> int { <usize as Step>::size(self.start, self.end) }
    fn done(self: UsizeRange) -> bool { <usize as Step>::size(self.start, self.end) <= 0 }
)]
impl Iterator for UsizeRange {
    type Item = usize;
    #[spec(
        fn(self: &mut UsizeRange[@old]) -> Option<usize[old.start]>[old.start < old.end]
        ensures self: UsizeRange{r: (old.start < old.end => r.start == <usize as Step>::step_forward(old.start, 1)) && r.end == old.end }
    )]
    fn next(&mut self) -> Option<usize> {
        if self.start >= self.end {
            None
        } else {
            let res = Some(self.start);
            self.start = self.start + 1;
            res
        }
    }
}

#[assoc(
    fn valid_item(self: I32Range, item: int) -> bool { self.start <= item && item < self.end }
    fn size(self: I32Range) -> int { <i32 as Step>::size(self.start, self.end) }
    fn done(self: I32Range) -> bool { <i32 as Step>::size(self.start, self.end) <= 0 }
)]
impl Iterator for I32Range {
    type Item = i32;
    #[spec(
        fn(self: &mut I32Range[@old]) -> Option<i32[old.start]>[old.start < old.end]
        ensures self: I32Range{r: (old.start < old.end => r.start == <i32 as Step>::step_forward(old.start, 1)) && r.end == old.end }
    )]
    fn next(&mut self) -> Option<i32> {
        if self.start >= self.end {
            None
        } else {
            let res = Some(self.start);
            self.start = self.start + 1;
            res
        }
    }
}

impl UsizeRange {
    #[trusted]
    #[spec(
        fn(Self[@s], f: F) -> UFMap[s]
        where
            F: FnMut(<UsizeRange as Iterator>::Item{item: <UsizeRange as Iterator>::valid_item(s, item)}) -> f64
    )]
    pub fn map_f64<'a, F: FnMut(<UsizeRange as Iterator>::Item) -> f64 + 'a>(self, f: F) -> UFMap<'a> {
        UFMap {
            iter: self,
            mapper: Box::new(f),
        }
    }

    #[trusted]
    #[spec(
        fn(Self[@s], f: F) -> URFMap[s]
        where
            F: FnMut(<UsizeRange as Iterator>::Item{item: <UsizeRange as Iterator>::valid_item(s, item)}) -> RVec<f64>
    )]
    pub fn map_rvec_f64<'a, F: FnMut(<UsizeRange as Iterator>::Item) -> RVec<f64> + 'a>(self, f: F) -> URFMap<'a> {
        URFMap {
            iter: self,
            mapper: Box::new(f),
        }
    }
}

#[refined_by(inner: UsizeRange)]
pub struct UFMap<'a> {
    #[field[UsizeRange[inner]]]
    iter: UsizeRange,
    mapper: Box<dyn FnMut(usize) -> f64 + 'a>,
}

#[assoc(
    fn size(x: UFMap) -> int { <UsizeRange as Iterator>::size(x.inner) }
    fn done(x: UFMap) -> bool { <UsizeRange as Iterator>::done(x.inner) }
    fn step(x: UFMap, y: UFMap) -> bool { <UsizeRange as Iterator>::step(x.inner, y.inner) }
)]
impl<'a> Iterator for UFMap<'a> {
    type Item = f64;
    #[trusted]
    #[spec(fn(self: &mut Self[@curr_s]) -> Option<f64>[!<Self as Iterator>::done(curr_s)]
           ensures self: Self{next_s: <Self as Iterator>::step(curr_s, next_s)})]
    fn next(&mut self) -> Option<f64> {
        self.iter.next()
            .map(|u| (self.mapper)(u))
    }
}

#[refined_by(inner: UsizeRange)]
pub struct URFMap<'a> {
    #[field[UsizeRange[inner]]]
    iter: UsizeRange,
    mapper: Box<dyn FnMut(usize) -> RVec<f64> + 'a>,
}

#[assoc(
    fn size(x: URFMap) -> int { <UsizeRange as Iterator>::size(x.inner) }
    fn done(x: URFMap) -> bool { <UsizeRange as Iterator>::done(x.inner) }
    fn step(x: URFMap, y: URFMap) -> bool { <UsizeRange as Iterator>::step(x.inner, y.inner) }
)]
impl<'a> Iterator for URFMap<'a> {
    type Item = RVec<f64>;
    #[trusted]
    #[spec(fn(self: &mut Self[@curr_s]) -> Option<RVec<f64>>[!<Self as Iterator>::done(curr_s)]
           ensures self: Self{next_s: <Self as Iterator>::step(curr_s, next_s)})]
    fn next(&mut self) -> Option<RVec<f64>> {
        self.iter.next()
            .map(|u| (self.mapper)(u))
    }
}
