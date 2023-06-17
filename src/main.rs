use ndarray_stats::QuantileExt;
// use rand::Rng;

use random_choice::random_choice;
use rayon::prelude::*;
use std::env;
use std::{vec};
use tsplib::{NodeCoord};

#[derive(Debug, Default)]
struct Graph {
    n_nodes: usize,
    distance_matrix: ndarray::Array2<f32>,
    intensity_matrix: ndarray::Array2<f32>,
}
impl Graph {
    pub fn new(n_nodes: usize, distance_matrix: ndarray::Array2<f32>) -> Self {
        let intensity_matrix = ndarray::Array2::ones((n_nodes, n_nodes));
        Self {
            n_nodes,
            distance_matrix,
            intensity_matrix,
        }
    }

    pub fn cycle_length(&self, cycle: &[usize]) -> f32 {
        let mut length = 0.0;
        for i in 0..(cycle.len() - 1) {
            length += self.distance_matrix[(
                cycle.get(i).unwrap().clone(),
                cycle.get(i + 1).unwrap().clone(),
            )];
        }
        length += self.distance_matrix[(
            cycle.last().unwrap().clone(),
            cycle.first().unwrap().clone(),
        )];
        length
    }

    pub fn reverse_segment_if_better(
        &self,
        cycle: &mut [usize],
        i: usize,
        j: usize,
        k: usize,
    ) -> f32 {
        let a = cycle[i - 1];
        let b = cycle[i];
        let c = cycle[j - 1];
        let d = cycle[j];
        let e = cycle[k - 1];
        let f = cycle[k % cycle.len()];

        let d0 = self.distance_matrix[(a, b)]
            + self.distance_matrix[(c, d)]
            + self.distance_matrix[(e, f)];
        let d1 = self.distance_matrix[(a, c)]
            + self.distance_matrix[(b, d)]
            + self.distance_matrix[(e, f)];
        let d2 = self.distance_matrix[(a, b)]
            + self.distance_matrix[(c, e)]
            + self.distance_matrix[(d, f)];
        let d3 = self.distance_matrix[(a, d)]
            + self.distance_matrix[(e, b)]
            + self.distance_matrix[(c, f)];
        let d4 = self.distance_matrix[(f, b)]
            + self.distance_matrix[(c, d)]
            + self.distance_matrix[(e, a)];

        if d0 > d1 {
            cycle[i..j].reverse();
            return -d0 + d1;
        } else if d0 > d2 {
            cycle[j..k].reverse();
            return -d0 + d2;
        } else if d0 > d4 {
            cycle[i..k].reverse();
            return -d0 + d4;
        } else if d0 > d3 {
            let mut tmp = vec![];
            for v in j..k {
                tmp.push(cycle[v]);
            }
            for v in i..j {
                tmp.push(cycle[v]);
            }
            for v in i..k {
                cycle[v] = tmp[v - i];
            }
            return -d0 + d3;
        }
        return 0.0;
    }

    pub fn three_opt(&self, cycle: &mut [usize]) {
        let n = cycle.len() - 1;
        let mut delta = 0.0;
        for a in 1..n {
            for b in a + 2..n {
                for c in b + 2..n + (a > 0) as usize {
                    delta += self.reverse_segment_if_better(cycle, a, b, c);
                }
            }
        }
        if delta >= 0.0 {
            println!("three_opt did not find improvement");
        }
    }

    pub fn ant_colony_optimization(&mut self, n_iterations: usize, ants_per_iteration: usize) {
        let mut best_cycle: Vec<_> = (0..self.n_nodes).collect();
        let mut best_len: f32 = f32::INFINITY;
        let mut old_best: f32 = best_len;
        let mut inertia = 0;
        let mean_dist = self.distance_matrix.mean().unwrap();

        for iteration in 0..n_iterations {
            println!("Iteration {}, best score is {}", iteration, best_len);

            // if iteration % 100 == 99 {
            //     self.break_most_use_edge();
            //     best_len *= 2.0;
            // }

            let mut generated_cycles: Vec<(Vec<usize>, f32)> = (0..ants_per_iteration)
                .into_iter()
                .into_par_iter() // Magic free performance
                .map(|_| self.traverse_graph(0))
                .collect();
            generated_cycles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // ignore top (worst) half
            // generated_cycles.truncate(generated_cycles.len() / 2);
            for _ in 0..2 {
                generated_cycles.push((best_cycle.clone(), best_len)); // Elitism
            }

            self.intensity_matrix *= 0.5; // pheromone decay
            for (i, (cycle, score)) in generated_cycles.iter().enumerate() {
                let d = (generated_cycles.len() - i) as f32 * mean_dist / score;
                for i in 1..self.n_nodes {
                    self.intensity_matrix[(cycle[i - 1], cycle[i])] += d;
                }
                self.intensity_matrix[(
                    cycle.last().unwrap().clone(),
                    cycle.first().unwrap().clone(),
                )] += d;
            }
            if generated_cycles.first().unwrap().1 < best_len {
                best_len = generated_cycles.first().unwrap().1;
                best_cycle = generated_cycles.first().unwrap().0.clone();
            }

            if old_best == best_len {
                inertia += 1;
            }

            old_best = best_len;
            if inertia > 20 {
                inertia = 0;
                self.intensity_matrix += self.intensity_matrix.mean().unwrap();
            }
            self.three_opt(&mut best_cycle);
            best_len = self.cycle_length(&best_cycle);
        }
        println!("Best found cycle was: {:?}", best_cycle);
    }

    pub fn traverse_graph(&self, _starting_node_ix: usize) -> (Vec<usize>, f32) {
        // let mut th_rng = rand::thread_rng();
        // let starting_node_ix = th_rng.gen_range(0..self.n_nodes);
        let starting_node_ix = fastrand::choice(0..self.n_nodes).unwrap();
        // println!("Using starting node : {}", starting_node_ix);
        let mut visited = vec![false; self.n_nodes];

        visited[starting_node_ix] = true;
        let mut cycle: Vec<usize> = vec![starting_node_ix];
        cycle.reserve(self.n_nodes);

        let mut jump_ix = vec![];
        let mut jump_val = vec![];
        let mut rng = random_choice();
        for _step in 0..self.n_nodes - 1 {
            jump_ix.clear();
            jump_val.clear();
            for node in 0..self.n_nodes {
                if visited[node] {
                    continue;
                }
                // we are only interested in theese nodes
                let current = cycle.last().unwrap().clone();
                let pheromone_intensity = self.intensity_matrix[(current, node)].max(0.0001);
                let v = (pheromone_intensity.powf(0.8))
                    / (self.distance_matrix[(current, node)].powf(3.5));
                jump_ix.push(node);
                jump_val.push(v);
            }
            let next = rng
                .random_choice_f32(&jump_ix, &jump_val, 1)
                .first()
                .unwrap()
                .to_owned()
                .to_owned();
            visited[next] = true;
            cycle.push(next);
        }
        let score = self.cycle_length(&cycle);
        (cycle, score)
    }

    pub fn break_most_use_edge(&mut self) {
        let ix = self.intensity_matrix.argmax().unwrap();
        self.distance_matrix[ix] *= 10.0;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage : {} <tsp instance>", args[0]);
        return;
    }
    let filename = &args[1];
    // let instance = tsplib::read("res/d198.tsp").unwrap();
    let instance = tsplib::read(filename).unwrap();
    // let instance = tsplib::read("res/att48.tsp").unwrap();
    // let instance = tsplib::read("res/berlin52.tsp").unwrap();
    // let instance = tsplib::read("res/att532.tsp").unwrap();
    // let instance = tsplib::read("res/u1060.tsp").unwrap();
    println!("Name: {}\nDimension: {}", instance.name, instance.dimension);

    // Load coordinates
    let node_index_coordinates = match instance.node_coord.unwrap() {
        NodeCoord::Two(coords) => coords,
        NodeCoord::Three(_) => todo!(),
    };
    let coordinates: Vec<_> = node_index_coordinates
        .iter()
        .map(|(_i, x, y)| (*x as f32, *y as f32))
        .collect();

    // Create cost matrix
    let mut cost_matrix = ndarray::Array2::<f32>::zeros((instance.dimension, instance.dimension));
    cost_matrix.indexed_iter_mut().for_each(|((a, b), e)| {
        *e = (coordinates[a].0 - coordinates[b].0).hypot(coordinates[a].1 - coordinates[b].1)
    });

    // Create a Graph
    println!("Cost matrix:\n{}", cost_matrix);
    let mut graph = Graph::new(instance.dimension, cost_matrix);
    graph.ant_colony_optimization(1000, 50);
}
