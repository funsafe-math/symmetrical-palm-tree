use random_choice::random_choice;
use std::{fs, io::Cursor, vec};
use tsplib::{read, EdgeWeightType, NodeCoord, Type};

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
            length += self
                .distance_matrix
                .get((
                    cycle.get(i).unwrap().clone(),
                    cycle.get(i + 1).unwrap().clone(),
                ))
                .unwrap();
        }
        length += self
            .distance_matrix
            .get((
                cycle.last().unwrap().clone(),
                cycle.first().unwrap().clone(),
            ))
            .unwrap();
        length
    }
    pub fn ant_colony_optimization(&mut self, n_iterations: usize, ants_per_iteration: usize) {
        let mut best_cycle: Vec<_> = (0..self.n_nodes).collect();
        let mut best_len: f32 = f32::INFINITY;
        let mean_dist = self.distance_matrix.mean().unwrap();

        for iteration in (0..n_iterations) {
            println!("Iteration {}, best score is {}", iteration, best_len);
            let mut generated_cycles: Vec<(Vec<usize>, f32)> = (0..ants_per_iteration)
                .into_iter()
                .map(|_| self.traverse_graph(0))
                .collect();
            generated_cycles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            // ignore top (worst) half
            // generated_cycles.truncate(generated_cycles.len() / 2);
            generated_cycles.push((best_cycle.clone(), best_len)); // Elitism

            self.intensity_matrix *= 0.95;
            for (cycle, score) in &generated_cycles {
                let d = mean_dist / score;
                for i in (1..self.n_nodes) {
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
        }
        println!("Best found cycle was: {:?}", best_cycle);
    }
    pub fn traverse_graph(&self, starting_node_ix: usize) -> (Vec<usize>, f32) {
        let mut visited = vec![false; self.n_nodes];
        visited[starting_node_ix] = true;
        let mut cycle: Vec<usize> = vec![starting_node_ix];
        cycle.reserve(self.n_nodes);

        for step in (0..self.n_nodes - 1) {
            let mut jump_ix = vec![];
            let mut jump_val = vec![];
            for node in (0..self.n_nodes) {
                if visited[node] {
                    continue;
                }
                // we are only interested in theese nodes
                let current = cycle.last().unwrap().clone();
                let pheromone_intensity = self.intensity_matrix[(current, node)].max(1.0e-3);
                let v = (pheromone_intensity.powf(0.9))
                    / (self.distance_matrix[(current, node)].powf(1.5));
                jump_ix.push(node);
                jump_val.push(v);
            }
            let next = random_choice()
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
}

fn main() {
    // let instance = tsplib::read("res/att48.tsp").unwrap();
    let instance = tsplib::read("res/att532.tsp").unwrap();
    println!("Name: {}\nDimension: {}", instance.name, instance.dimension);

    // Load coordinates
    let node_index_coordinates = match instance.node_coord.unwrap() {
        NodeCoord::Two(coords) => coords,
        NodeCoord::Three(_) => todo!(),
    };
    let coordinates: Vec<_> = node_index_coordinates
        .iter()
        .map(|(i, x, y)| (x, y))
        .collect();

    // Create cost matrix
    let mut cost_matrix = ndarray::Array2::<f32>::zeros((instance.dimension, instance.dimension));
    cost_matrix.indexed_iter_mut().for_each(|((x, y), e)| {
        *e = (coordinates[x].0 - coordinates[y].0).hypot(coordinates[x].1 - coordinates[y].1)
    });

    // Create a Graph
    println!("Cost matrix:\n{}", cost_matrix);
    let mut graph = Graph::new(instance.dimension, cost_matrix);
    graph.ant_colony_optimization(10_000, 500);
}
