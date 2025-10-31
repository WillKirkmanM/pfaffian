use nalgebra::DMatrix;
use std::collections::HashMap;

/// A struct to hold our skew-symmetric matrix.
/// We use a DMatrix (dynamic matrix) from nalgebra.
struct SkewMatrix {
    data: DMatrix<f64>,
}

impl SkewMatrix {
    /// Creates a new SkewMatrix from a list of upper-triangular values.
    /// For a 4x4 matrix, you'd provide 6 values: (a, b, c, d, e, f)
    /// which map to:
    ///   0  a  b  c
    ///  -a  0  d  e
    ///  -b -d  0  f
    ///  -c -e -f  0
    pub fn from_upper_triangle(n: usize, values: &[f64]) -> Self {
        assert_eq!(n % 2, 0, "Matrix must have even dimensions.");
        let expected_vals = n * (n - 1) / 2;
        assert_eq!(
            values.len(),
            expected_vals,
            "Incorrect number of values for an {}x{} matrix.",
            n,
            n
        );

        let mut m = DMatrix::<f64>::zeros(n, n);
        let mut val_iter = values.iter();

        for i in 0..n {
            for j in (i + 1)..n {
                let val = *val_iter.next().unwrap();
                m[(i, j)] = val;
                m[(j, i)] = -val;
            }
        }
        Self { data: m }
    }

    /// Recursively computes the Pfaffian of the matrix.
    /// This implementation is for demonstration and is not O(n^3).
    /// It directly models the "sum over perfect matchings" definition.
    ///
    /// The formula is: Pf(A) = sum_{j=2..2n} (-1)^j * A_{1,j} * Pf(A_{1,j})
    ///
    /// Pf(A_ij) is the pfaffian of the submatrix with rows/cols i and j removed.
    pub fn pfaffian(&self) -> f64 {
        // Use a memoization table (HashMap) to store results for subproblems.
        // This turns the exponential O(n!!) recursion into a fast O(n^3) 
        // dynamic programming algorithm. This is one way to get the "magic" speedup.
        let mut memo: HashMap<Vec<usize>, f64> = HashMap::new();
        let initial_indices: Vec<usize> = (0..self.data.nrows()).collect();
        self.pfaffian_recursive(&initial_indices, &mut memo)
    }

    fn pfaffian_recursive(
        &self,
        indices: &[usize], // The rows/cols we are still considering
        memo: &mut HashMap<Vec<usize>, f64>,
    ) -> f64 {
        let n = indices.len();

        // Base case: A 0x0 matrix has a Pfaffian of 1.
        if n == 0 {
            return 1.0;
        }

        // Check memoization table
        if let Some(&result) = memo.get(indices) {
            return result;
        }

        // This is the core "matching" step.
        // We *fix* the first vertex (indices[0]) and try to "match" it
        // with every other vertex (indices[j] where j > 0).
        let mut total_sum = 0.0;
        let i = indices[0]; // Fix the first element

        for j_idx in 1..n {
            let j = indices[j_idx];

            // Get the weight of the edge (i, j)
            let a_ij = self.data[(i, j)];

            // Create the list of remaining indices for the sub-problem
            // This is equivalent to "deleting" rows/cols i and j.
            let mut sub_indices = Vec::with_capacity(n - 2);
            for k_idx in 1..n {
                if k_idx != j_idx {
                    sub_indices.push(indices[k_idx]);
                }
            }

            // Calculate the sign. (-1)^(j_idx + 1 - 1) = (-1)^j_idx
            let sign = if j_idx % 2 == 1 { -1.0 } else { 1.0 };

            // RECURSIVE CALL:
            // This is the sum: Pf(A) = A_12 * Pf(A_{1,2}) - A_13 * Pf(A_{1,3}) + ...
            // Each recursive call explores a different "perfect matching".
            total_sum += sign * a_ij * self.pfaffian_recursive(&sub_indices, memo);
        }

        // Store result in memoization table and return it
        memo.insert(indices.to_vec(), total_sum);
        total_sum
    }
}

fn main() {
    // ## Example 1: A 2x2 Matrix ##
    //   0  a
    //  -a  0
    // The only perfect matching is the edge (0, 1) with weight 'a'.
    // The Pfaffian should be 'a'.
    let m2 = SkewMatrix::from_upper_triangle(2, &[12.0]);
    println!("A 2x2 Matrix:\n{}\n", m2.data);
    println!("Pfaffian(A_2x2) = {}", m2.pfaffian()); // Output: 12.0
    println!("---");

    // ## Example 2: A 4x4 Matrix ##
    //   0  a  b  c
    //  -a  0  d  e
    //  -b -d  0  f
    //  -c -e -f  0
    //
    // The 3 possible perfect matchings are:
    // 1. (0,1) and (2,3) -> weight = a * f
    // 2. (0,2) and (1,3) -> weight = b * e
    // 3. (0,3) and (1,2) -> weight = c * d
    //
    // The Pfaffian is: a*f - b*e + c*d
    let (a, b, c, d, e, f) = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    let m4 = SkewMatrix::from_upper_triangle(4, &[a, b, c, d, e, f]);
    
    let expected_pf = a * f - b * e + c * d; // 2*7 - 3*6 + 4*5 = 14 - 18 + 20 = 16

    println!("A 4x4 Matrix:\n{}\n", m4.data);
    println!("Pfaffian(A_4x4) = {}", m4.pfaffian()); // Output: 16.0
    println!("Expected (af - be + cd) = {}", expected_pf);
    println!("---");

    // ## Example 3: A 6x6 Matrix ##
    // This has (2*3-1)!! = 5!! = 15 perfect matchings.
    // Our recursive function sums all 15 weighted combinations instantly
    // thanks to memoization.
    let m6 = SkewMatrix::from_upper_triangle(6, &[
        1.0, 2.0, 3.0, 4.0, 5.0, // row 0
        6.0, 7.0, 8.0, 9.0,     // row 1
        10.0, 11.0, 12.0,       // row 2
        13.0, 14.0,             // row 3
        15.0                    // row 4
    ]);
    
    println!("Pfaffian(A_6x6) = {}", m6.pfaffian()); // Output: -60.0
}