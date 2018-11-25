use mpi::{collective::SystemOperation, traits::*};
use rayon::prelude::*;

fn f(a: f64) -> f64 {
    4.0 / (1. + a * a)
}

fn main() {
    let universe = mpi::initialize().expect("Did not initialize MPI");

    let comm = universe.world();

    let rank = comm.rank();
    let size = comm.size();

    println!("{}: Num threads = {}", rank, rayon::current_num_threads());

    let root_process = comm.process_at_rank(0);

    let mut n = if rank == root_process.rank() {
        let args: Vec<_> = std::env::args().collect();

        if args.len() <= 1 {
            100
        } else {
            match args[1].parse::<i64>() {
                Ok(x) => x,
                Err(_) => {
                    eprintln!("First argument must be an integer!");
                    std::process::exit(1);
                }
            }
        }
    } else {
        0
    };

    root_process.broadcast_into(&mut n);

    let h = 1.0 / (n as f64);

    let size64 = size as i64;
    let rank64 = rank as i64;

    let num_local = n / size64 + if rank64 < n % size64 { 1 } else { 0 };

    let partial_pi = if rayon::current_num_threads() > 1 {
        const BATCH: i64 = 2000;

        let num_batches = num_local / BATCH;
        let num_remain = num_local % BATCH;

        (0..num_batches)
            .into_par_iter()
            // .flat_map(|x| x * BATCH..(x + 1) * BATCH)
            .map(|x| {
                (x * BATCH..(x + 1) * BATCH)
                    .map(|i| f(h * (((i * size64 + rank64 + 1) as f64) - 0.5)))
                    .sum::<f64>()
            })
            .sum::<f64>()
            + if num_remain != 0 {
                (num_batches * BATCH..num_local)
                    .map(|i| f(h * (((i * size64 + rank64 + 1) as f64) - 0.5)))
                    .sum::<f64>()
            } else {
                0.0
            }
    } else {
        (0..num_local)
            .map(|i| f(h * (((i * size64 + rank64 + 1) as f64) - 0.5)))
            .sum::<f64>()
    } * h;

    if rank == root_process.rank() {
        let mut pi = 0.;
        root_process.reduce_into_root(&partial_pi, &mut pi, SystemOperation::sum());

        println!("PI is something like: {}", pi);
    } else {
        root_process.reduce_into(&partial_pi, SystemOperation::sum());
    }
}
