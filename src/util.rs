#[derive(Debug, Clone)]
pub struct Permutations(Vec<usize>);

impl Permutations {
    pub fn new(size: usize) -> Self {
        Self((0..size).collect())
    }
}

impl Iterator for Permutations {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let size = self.0.len();
        if size == 0 {
            self.0 = vec![1];
            return Some(Vec::new());
        }
        if size == 1 {
            if self.0[0] == 0 {
                self.0[0] = 1;
                return Some(vec![0]);
            } else {
                return None;
            }
        }

        // Find the longest decreasing suffix
        let suffix_start = 'search: {
            let mut prev = *self.0.last().unwrap();
            for i in (0..size - 1).rev() {
                if self.0[i] < prev {
                    break 'search i + 1;
                }
                prev = self.0[i];
            }

            0
        };

        if suffix_start == 0 {
            return Some(std::mem::replace(&mut self.0, vec![1]));
        }

        let res = self.0.clone();
        self.0[suffix_start..].reverse();
        // Find smallest element element that's larger than the element proceeding suffix
        for i in suffix_start..size {
            if self.0[i] > self.0[suffix_start - 1] {
                self.0.swap(suffix_start - 1, i);
                return Some(res);
            }
        }

        unreachable!()
    }
}

pub struct CartesianProduct<I1: Iterator, I2: Iterator> {
    first_item: Option<I1::Item>,
    first_iter: I1,
    second_start: I2,
    second_iter: I2,
}

pub fn cartesian_product<I1: Iterator, I2: Iterator>(
    mut first_iter: I1,
    second_iter: I2,
) -> CartesianProduct<I1, I2>
where
    I1::Item: Clone,
    I2: Clone,
{
    CartesianProduct {
        first_item: first_iter.next(),
        first_iter,
        second_start: second_iter.clone(),
        second_iter: second_iter,
    }
}

impl<I1: Iterator, I2: Iterator> Iterator for CartesianProduct<I1, I2>
where
    I1::Item: Clone,
    I2: Clone,
{
    type Item = (I1::Item, I2::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first_item = self.first_item.as_ref()?;

        if let Some(second_item) = self.second_iter.next() {
            return Some((first_item.clone(), second_item));
        }
        self.second_iter = self.second_start.clone();
        self.first_item = self.first_iter.next();
        self.next()
    }
}

/// Computes greatest common divisor of given numbers.
pub fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
/* pub fn all_mappings<T1: Eq + Clone + std::hash::Hash, T2: Clone + Copy>(choices: Vec<(T1, Vec<T2>)>) -> Vec<HashMap<T1, T2>> {
    let (current, current_choices) = &choices[0];
    if choices.len() == 1 {
        return current_choices.into_iter().map( |choice| HashMap::from([(current.clone(), *choice)]) ).collect();
    }

    let mut result = Vec::new();
    let next = all_mappings(choices.clone()[1..].to_vec());

    if current_choices.len() == 0 {
        return next
    }
    if next.len() == 0 {
        return current_choices.into_iter().map( |choice| HashMap::from([(current.clone(), *choice)]) ).collect();
    }

    for current_choice in current_choices {
        let mappings = next.clone().into_iter().map( |hash_map| { let mut clone = hash_map.clone(); clone.insert(current.clone(), *current_choice); clone } );
        result.extend(mappings);
    }
    result
} */

/// Generates all possible mappings from T1 to T2 given all the possible choices of T2 for all T1.
pub struct AllMappings<T1: Eq + Clone + Hash, T2: Clone + Copy> {
    choices: Vec<(T1, Vec<T2>)>,
    indices: Vec<usize>,
    finished: bool,
}

pub fn all_mappings<T1: Eq + Clone + Hash, T2: Clone + Copy>(
    choices: Vec<(T1, Vec<T2>)>,
) -> AllMappings<T1, T2> {
    AllMappings {
        indices: vec![0; choices.len()],
        choices: choices,
        finished: false,
    }
}

impl<T1: Eq + Clone + Hash, T2: Clone + Copy> AllMappings<T1, T2> {
    fn next_indices(&self) -> Vec<usize> {
        let mut i = self.indices.len() - 1;
        let mut stop = false;
        let mut indices = self.indices.clone();
        while !stop {
            indices[i] += 1;
            if indices[i] >= self.choices[i].1.len() {
                indices[i] = 0;
            } else {
                stop = true;
            }
            if i == 0 {
                stop = true;
            } else {
                i -= 1;
            }
        }
        indices
    }
}

impl<T1: Eq + Clone + Hash, T2: Clone + Copy> Iterator for AllMappings<T1, T2> {
    type Item = HashMap<T1, T2>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut result: HashMap<T1, T2> = HashMap::new();

        for i in 0..self.indices.len() {
            let (current, choices) = &self.choices[i];
            if choices.len() > 0 {
                let choice = choices[self.indices[i]];
                result.insert(current.clone(), choice);
            }
        }

        let next_indices = self.next_indices();
        self.finished = (0..self.indices.len()).all(|i| next_indices[i] == 0);
        self.indices = next_indices;

        Some(result)
    }
}

pub fn combinations_of_len<T1: Clone + Copy>(elements: &Vec<T1>, length: u32) -> Vec<Vec<T1>> {
    if length <= 1 {
        return elements.clone().into_iter().map(|is| vec![is]).collect();
    }

    let mut result = Vec::new();
    let prev_combinations = combinations_of_len(elements, length - 1);
    for symbol in elements {
        let mut extended_combinations = prev_combinations.clone();
        for comb in &mut extended_combinations {
            comb.push(*symbol)
        }
        result.extend(extended_combinations);
    }
    return result;
}

pub fn strongly_connected_components<T: Eq + Hash + Clone>(
    edges: impl Iterator<Item = (T, T)>,
) -> Vec<HashSet<T>> {
    let mut elements = HashSet::<T>::new();
    let mut neighbors = HashMap::<T, HashSet<T>>::new();
    for edge in edges {
        let (u, v): (T, T) = edge;
        elements.insert(u.clone());
        elements.insert(v.clone());
        if !neighbors.contains_key(&u) {
            neighbors.insert(u.clone(), HashSet::<T>::new());
        }
        neighbors.get_mut(&u).unwrap().insert(v.clone());
    }

    // Tarjan's strongly connected components algorithm
    struct RecursionEnv<T: Eq + Hash + Clone> {
        result: Vec<HashSet<T>>,

        index_of: HashMap<T, u32>,
        lowlink_of: HashMap<T, u32>,
        is_on_stack: HashSet<T>,

        index: u32,
        stack: Vec<T>,
    }

    let mut env = RecursionEnv {
        result: vec![],

        index_of: HashMap::new(),
        lowlink_of: HashMap::new(),
        is_on_stack: HashSet::new(),

        index: 0u32,
        stack: vec![],
    };

    for v in elements {
        if !env.index_of.contains_key(&v) {
            strongconnect(&v, &mut env, &neighbors);
        }
    }

    fn strongconnect<T: Eq + Hash + Clone>(
        v: &T,
        env: &mut RecursionEnv<T>,
        neighbors: &HashMap<T, HashSet<T>>,
    ) {
        env.index_of.insert(v.clone(), env.index);
        env.lowlink_of.insert(v.clone(), env.index);
        env.index += 1;
        env.stack.push(v.clone());
        env.is_on_stack.insert(v.clone());

        if let Some(neigh) = neighbors.get(v) {
            for w in neigh.iter() {
                if !env.index_of.contains_key(&w) {
                    strongconnect(w, env, neighbors);
                    let min = std::cmp::min(
                        *env.lowlink_of.get(v).unwrap(),
                        *env.lowlink_of.get(w).unwrap(),
                    );
                    env.lowlink_of.insert(v.clone(), min);
                } else if env.is_on_stack.contains(w) {
                    let min = std::cmp::min(
                        *env.lowlink_of.get(v).unwrap(),
                        *env.index_of.get(w).unwrap(),
                    );
                    env.lowlink_of.insert(v.clone(), min);
                }
            }
        }

        if env.lowlink_of.get(v).unwrap() == env.index_of.get(v).unwrap() {
            let mut component = HashSet::<T>::new();

            env.is_on_stack.remove(v);
            component.insert(v.clone());

            let mut w = env.stack.pop().unwrap();
            while w != *v {
                env.is_on_stack.remove(&w);
                component.insert(w);
                w = env.stack.pop().unwrap();
            }

            env.result.push(component);
        }
    }

    return env.result;
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_permutations_count() {
        assert_eq!(Permutations::new(0).count(), 1);
        assert_eq!(Permutations::new(1).count(), 1);
        assert_eq!(Permutations::new(2).count(), 2);
        assert_eq!(Permutations::new(3).count(), 6);
        assert_eq!(Permutations::new(4).count(), 24);
        assert_eq!(Permutations::new(5).count(), 120);
        assert_eq!(Permutations::new(6).count(), 720);
    }

    #[test]
    fn test_permutations_uniqueness() {
        fn count_unique(n: usize) -> usize {
            let perm: HashSet<_> = Permutations::new(n).collect();
            return perm.len();
        }
        assert_eq!(count_unique(0), 1);
        assert_eq!(count_unique(1), 1);
        assert_eq!(count_unique(2), 2);
        assert_eq!(count_unique(3), 6);
        assert_eq!(count_unique(4), 24);
        assert_eq!(count_unique(5), 120);
        assert_eq!(count_unique(6), 720);
    }

    #[test]
    fn test_permutations_order() {
        fn is_sorted<Item: Ord>(mut iter: impl Iterator<Item = Item>) -> bool {
            let Some(mut prev) = iter.next() else {
                return true;
            };
            for i in iter {
                if i <= prev {
                    return false;
                }
                prev = i;
            }
            true
        }
        assert!(is_sorted(Permutations::new(0)));
        assert!(is_sorted(Permutations::new(1)));
        assert!(is_sorted(Permutations::new(2)));
        assert!(is_sorted(Permutations::new(3)));
        assert!(is_sorted(Permutations::new(4)));
        assert!(is_sorted(Permutations::new(5)));
        assert!(is_sorted(Permutations::new(6)));
    }

    #[test]
    fn test_cartesian_product() {
        assert_eq!(
            cartesian_product(0..3, 5..7).collect::<Vec<_>>(),
            vec![(0, 5), (0, 6), (1, 5), (1, 6), (2, 5), (2, 6)]
        );
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(0, 1), 1);
        assert_eq!(gcd(0, 2), 2);
        assert_eq!(gcd(0, 3), 3);
        assert_eq!(gcd(1, 0), 1);
        assert_eq!(gcd(2, 0), 2);
        assert_eq!(gcd(3, 0), 3);
        assert_eq!(gcd(5, 7), 1);
        assert_eq!(gcd(6, 9), 3);
        assert_eq!(gcd(9, 6), 3);
    }

    #[test]
    fn test_all_mappings() {
        assert_eq!(
            all_mappings(vec![(0, vec![0, 1, 2]), (1, vec![0, 1]), (2, vec![0])])
                .collect::<Vec<_>>(),
            vec![
                HashMap::from([(0, 0), (1, 0), (2, 0)]),
                HashMap::from([(0, 0), (1, 1), (2, 0)]),
                HashMap::from([(0, 1), (1, 0), (2, 0)]),
                HashMap::from([(0, 1), (1, 1), (2, 0)]),
                HashMap::from([(0, 2), (1, 0), (2, 0)]),
                HashMap::from([(0, 2), (1, 1), (2, 0)])
            ]
        );

        assert_eq!(
            all_mappings(vec![(0, vec![0]), (1, vec![0, 1]), (2, vec![0, 1, 2])])
                .collect::<Vec<_>>(),
            vec![
                HashMap::from([(0, 0), (1, 0), (2, 0)]),
                HashMap::from([(0, 0), (1, 0), (2, 1)]),
                HashMap::from([(0, 0), (1, 0), (2, 2)]),
                HashMap::from([(0, 0), (1, 1), (2, 0)]),
                HashMap::from([(0, 0), (1, 1), (2, 1)]),
                HashMap::from([(0, 0), (1, 1), (2, 2)])
            ]
        );

        assert_eq!(
            all_mappings(vec![(0, vec![0, 1]), (1, vec![0, 1]), (2, vec![0, 1])])
                .collect::<Vec<_>>(),
            vec![
                HashMap::from([(0, 0), (1, 0), (2, 0)]),
                HashMap::from([(0, 0), (1, 0), (2, 1)]),
                HashMap::from([(0, 0), (1, 1), (2, 0)]),
                HashMap::from([(0, 0), (1, 1), (2, 1)]),
                HashMap::from([(0, 1), (1, 0), (2, 0)]),
                HashMap::from([(0, 1), (1, 0), (2, 1)]),
                HashMap::from([(0, 1), (1, 1), (2, 0)]),
                HashMap::from([(0, 1), (1, 1), (2, 1)])
            ]
        );

        assert_eq!(
            all_mappings(vec![(0, vec![0, 2]), (1, vec![]), (2, vec![2, 0])]).collect::<Vec<_>>(),
            vec![
                HashMap::from([(2, 2), (0, 0)]),
                HashMap::from([(2, 0), (0, 0)]),
                HashMap::from([(2, 2), (0, 2)]),
                HashMap::from([(2, 0), (0, 2)])
            ]
        );

        assert_eq!(
            all_mappings(vec![(0, vec![0]), (1, vec![0]), (2, vec![])]).collect::<Vec<_>>(),
            vec![HashMap::from([(1, 0), (0, 0)])]
        );
    }

    #[test]
    fn test_combinations_of_len() {
        let elements = vec![0, 1, 2, 3];

        assert_eq!(
            combinations_of_len(&elements, 1),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );

        assert_eq!(
            combinations_of_len(&elements, 2),
            vec![
                vec![0, 0],
                vec![1, 0],
                vec![2, 0],
                vec![3, 0],
                vec![0, 1],
                vec![1, 1],
                vec![2, 1],
                vec![3, 1],
                vec![0, 2],
                vec![1, 2],
                vec![2, 2],
                vec![3, 2],
                vec![0, 3],
                vec![1, 3],
                vec![2, 3],
                vec![3, 3]
            ]
        );

        assert_eq!(
            combinations_of_len(&elements, 3),
            vec![
                vec![0, 0, 0],
                vec![1, 0, 0],
                vec![2, 0, 0],
                vec![3, 0, 0],
                vec![0, 1, 0],
                vec![1, 1, 0],
                vec![2, 1, 0],
                vec![3, 1, 0],
                vec![0, 2, 0],
                vec![1, 2, 0],
                vec![2, 2, 0],
                vec![3, 2, 0],
                vec![0, 3, 0],
                vec![1, 3, 0],
                vec![2, 3, 0],
                vec![3, 3, 0],
                vec![0, 0, 1],
                vec![1, 0, 1],
                vec![2, 0, 1],
                vec![3, 0, 1],
                vec![0, 1, 1],
                vec![1, 1, 1],
                vec![2, 1, 1],
                vec![3, 1, 1],
                vec![0, 2, 1],
                vec![1, 2, 1],
                vec![2, 2, 1],
                vec![3, 2, 1],
                vec![0, 3, 1],
                vec![1, 3, 1],
                vec![2, 3, 1],
                vec![3, 3, 1],
                vec![0, 0, 2],
                vec![1, 0, 2],
                vec![2, 0, 2],
                vec![3, 0, 2],
                vec![0, 1, 2],
                vec![1, 1, 2],
                vec![2, 1, 2],
                vec![3, 1, 2],
                vec![0, 2, 2],
                vec![1, 2, 2],
                vec![2, 2, 2],
                vec![3, 2, 2],
                vec![0, 3, 2],
                vec![1, 3, 2],
                vec![2, 3, 2],
                vec![3, 3, 2],
                vec![0, 0, 3],
                vec![1, 0, 3],
                vec![2, 0, 3],
                vec![3, 0, 3],
                vec![0, 1, 3],
                vec![1, 1, 3],
                vec![2, 1, 3],
                vec![3, 1, 3],
                vec![0, 2, 3],
                vec![1, 2, 3],
                vec![2, 2, 3],
                vec![3, 2, 3],
                vec![0, 3, 3],
                vec![1, 3, 3],
                vec![2, 3, 3],
                vec![3, 3, 3]
            ]
        );
    }

    #[test]
    fn test_strongly_connected_components() {
        fn vec_of_sets_eq<T: Eq + Hash>(v1: Vec<HashSet<T>>, v2: Vec<HashSet<T>>) -> bool {
            if v1.len() != v2.len() {
                return false;
            }

            for u in v1 {
                if !v2.contains(&u) {
                    return false;
                }
            }

            true
        }

        let edges1: Vec<(u32, u32)> = vec![];
        let components1 = strongly_connected_components(edges1.into_iter());
        let answer1 = vec![];
        assert!(vec_of_sets_eq(components1, answer1));

        let edges2 = vec![(0, 0)];
        let components2 = strongly_connected_components(edges2.into_iter());
        let answer2 = vec![HashSet::from([0])];
        assert!(vec_of_sets_eq(components2, answer2));

        let edges3 = vec![(0, 1), (1, 0)];
        let components3 = strongly_connected_components(edges3.into_iter());
        let answer3 = vec![HashSet::from([0, 1])];
        assert!(vec_of_sets_eq(components3, answer3));

        let edges4 = vec![(0, 1), (1, 0), (2, 2)];
        let components4 = strongly_connected_components(edges4.into_iter());
        let answer4 = vec![HashSet::from([0, 1]), HashSet::from([2])];
        assert!(vec_of_sets_eq(components4, answer4));

        let edges5 = vec![(0, 0), (0, 1), (1, 1)];
        let components5 = strongly_connected_components(edges5.into_iter());
        let answer5 = vec![HashSet::from([0]), HashSet::from([1])];
        assert!(vec_of_sets_eq(components5, answer5));

        let edges6 = vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
        let components6 = strongly_connected_components(edges6.into_iter());
        let answer6 = vec![HashSet::from([0, 1, 2])];
        assert!(vec_of_sets_eq(components6, answer6));

        let edges7 = vec![
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (0, 3),
            (3, 4),
            (4, 3),
        ];
        let components7 = strongly_connected_components(edges7.into_iter());
        let answer7 = vec![HashSet::from([0, 1, 2]), HashSet::from([3, 4])];
        assert!(vec_of_sets_eq(components7, answer7));

        let edges8 = vec![
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (0, 3),
            (3, 4),
            (4, 3),
            (4, 0),
        ];
        let components8 = strongly_connected_components(edges8.into_iter());
        let answer8 = vec![HashSet::from([0, 1, 2, 3, 4])];
        assert!(vec_of_sets_eq(components8, answer8));
    }
}
