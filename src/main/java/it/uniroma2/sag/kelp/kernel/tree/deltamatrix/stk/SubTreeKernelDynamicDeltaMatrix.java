package it.uniroma2.sag.kelp.kernel.tree.deltamatrix.stk;

import gnu.trove.map.hash.TIntFloatHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import it.uniroma2.sag.kelp.kernel.tree.deltamatrix.DeltaMatrix;

public class SubTreeKernelDynamicDeltaMatrix implements DeltaMatrix {


	/**
	 * Sparse implementation of a matrix
	 */
	private TIntObjectHashMap<TIntFloatHashMap> matrix;

	public SubTreeKernelDynamicDeltaMatrix() {
		matrix = new TIntObjectHashMap<TIntFloatHashMap>();
	}

	/**
	 * Insert a value in the matrix
	 * 
	 * @param i
	 *            row index
	 * @param j
	 *            column index
	 * @param v
	 *            value to insert in delta_matrix[i][j]
	 */
	public void add(int i, int j, float v) {
		if (!matrix.containsKey(i))
			matrix.put(i, new TIntFloatHashMap());
		matrix.get(i).put(j, v);
	}

	/**
	 * Get a value from the matrix
	 * 
	 * @param i
	 *            row index
	 * @param j
	 *            column index
	 * @return value to retrieve from the delta_matrix[[i][j]
	 */
	public float get(int i, int j) {
		if (!matrix.containsKey(i))
			return 0;
		return matrix.get(i).get(j);
	}

	/**
	 * Clear the delta matrix
	 */
	public void clear() {
		matrix.clear();
	}
}
