/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.kernel.tree;

import it.uniroma2.sag.kelp.data.representation.tree.TreeRepresentation;
import it.uniroma2.sag.kelp.data.representation.tree.node.TreeNode;
import it.uniroma2.sag.kelp.data.representation.tree.node.TreeNodePairs;
import it.uniroma2.sag.kelp.kernel.DirectKernel;
import it.uniroma2.sag.kelp.kernel.tree.deltamatrix.DeltaMatrix;
import it.uniroma2.sag.kelp.kernel.tree.deltamatrix.StaticDeltaMatrix;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * SubTree Kernel implementation.
 * 
 * A SubTree Kernel is a convolution kernel. The kernel function is defined as:
 * </br>
 * 
 * \(K(T_1,T_2) = \sum_{n_1 \in N_{T_1}} \sum_{n_2 \in N_{T_2}}
 * \Delta(n_1,n_2)\)
 * 
 * </br>
 * 
 * where \(\Delta(n_1,n_2)\) can be computed as:</br> - if productions at
 * \(n_1\) and \(n_2\) are different then \(\Delta(n_1,n_2)=0\)</br> - if the
 * productions at n1 and n2 are the same, and \(n_1\) and \(n_2\) have only leaf
 * children then \(\Delta(n_1,n_2)=\lambda\)</br> - if the productions at n1 and
 * n2 are the same, and \(n_1\) and \(n_2\) are not pre-terminals then</br>
 * \(\Delta(n_1,n_2)=\lambda \prod_{j=1}^{nc(n_1)} (\sigma + \Delta(c_{n_1}^j,
 * c_{n_2}^j))\)
 * 
 * </br></br> For more details see [Vishwanathan and Smola, 2001; Moschitti,
 * EACL 2006]
 * 
 * [Vishwanathan and Smola, 2002], S.V.N. Vishwanathan and A.J. Smola. Fast
 * kernels on strings and trees. In Proceedings of Neural Information Processing
 * Systems, 2002.
 * 
 * [Moschitti, EACL2006] Alessandro Moschitti. Making Tree Kernels Practical for
 * Natural Language Learning EACL, (2006)
 * 
 * @author Danilo Croce, Giuseppe Castellucci
 */

@JsonTypeName("stk")
public class SubTreeKernel extends DirectKernel<TreeRepresentation> {

	/**
	 * Decay factor
	 */
	private float lambda;

	/**
	 * The delta matrix, used to cache the delta functions applied to subtrees
	 */
	private DeltaMatrix deltaMatrix;

	/**
	 * This value is used to mark node pairs in the delta_matrix that have not
	 * been evaluated
	 */
	private static final int NO_RESPONSE = -1;

	/**
	 * SubTree Kernel
	 * 
	 * @param lambda
	 *            Decay Factor
	 * @param representationIdentifier
	 *            Identifier of the Tree representation on which the kernel
	 *            works
	 */
	public SubTreeKernel(float lambda, String representationIdentifier) {
		super(representationIdentifier);
		this.lambda = lambda;
		this.deltaMatrix = new StaticDeltaMatrix();
	}

	/**
	 * SubTree Kernel constructor. It uses lambda=0.4
	 * 
	 * @param representationIdentifier
	 *            Identifier of the Tree representation on which the kernel
	 *            works
	 */
	public SubTreeKernel(String representationIdentifier) {
		this(0.4f, representationIdentifier);
	}

	/**
	 * SubTree Kernel: default constructor. It should not be used, please use
	 * SubTreeKernel(String) or SubTreeKernel(float,String). This is only used
	 * by the json serializer/deserializer.
	 */
	public SubTreeKernel() {
		this(0.4f, "0");
	}

	/**
	 * Get the decay factor
	 * 
	 * @return the decay factor
	 */
	public float getLambda() {
		return lambda;
	}

	/**
	 * Set the decay factor
	 * 
	 * @param lambda
	 *            the decay factor
	 */
	public void setLambda(float lambda) {
		this.lambda = lambda;
	}

	/**
	 * SubTree Kernel Delta Function
	 * 
	 * @param Nx
	 *            root of the first tree
	 * @param Nz
	 *            root of the second tree
	 * @return
	 */
	private float stkDeltaFunction(TreeNode Nx, TreeNode Nz) {
		if (deltaMatrix.get(Nx.getId(), Nz.getId()) != NO_RESPONSE)
			return deltaMatrix.get(Nx.getId(), Nz.getId()); // cashed
		else {
			float prod = 1;
			ArrayList<TreeNode> NxChildren = Nx.getChildren();
			ArrayList<TreeNode> NzChildren = Nz.getChildren();
			for (int i = 0; i < NxChildren.size() && i < NzChildren.size(); i++) {

				if (NxChildren.get(i).hasChildren()
						&& NzChildren.get(i).hasChildren()
						&& NxChildren.get(i).getProduction()
								.equals(NzChildren.get(i).getProduction())) {

					prod *= 1.0 + stkDeltaFunction(NxChildren.get(i),
							NzChildren.get(i));
				}
			}
			deltaMatrix.add(Nx.getId(), Nz.getId(), lambda * prod);
			return lambda * prod;
		}
	}

	/**
	 * Evaluate the Subtree Tree Kernel
	 * 
	 * @param a
	 *            First tree
	 * @param b
	 *            Second Tree
	 * @return Kernel value
	 */
	private float evaluateKernelNotNormalize(TreeRepresentation a,
			TreeRepresentation b) {

		float k = 0;

		// Initialize the delta function cache
		deltaMatrix.clear();

		// Determine the subtrees whose root have the same label. This
		// optimization has been proposed in [Moschitti, EACL 2006].
		ArrayList<TreeNodePairs> pairs = determineSubList(a, b);

		// Estimate the kernel function
		for (int i = 0; i < pairs.size(); i++) {

			float deltaValue = stkDeltaFunction(pairs.get(i).getNx(), pairs
					.get(i).getNz());

			k += deltaValue;
		}

		return k;
	}

	/**
	 * Determine the subtrees (from the two trees) whose root have the same
	 * label. This optimization has been proposed in [Moschitti, EACL 2006].
	 * 
	 * @param a
	 *            First Tree
	 * @param b
	 *            Second Tree
	 * @return The node pairs having the same label
	 */
	private ArrayList<TreeNodePairs> determineSubList(TreeRepresentation a,
			TreeRepresentation b) {

		ArrayList<TreeNodePairs> intersect = new ArrayList<TreeNodePairs>();

		int i = 0, j = 0, j_old, j_final;
		int n_a, n_b;
		int cfr;

		List<TreeNode> nodesA = a.getOrderedNodeSetByProduction();
		List<TreeNode> nodesB = b.getOrderedNodeSetByProduction();

		n_a = nodesA.size();
		n_b = nodesB.size();

		while (i < n_a && j < n_b) {

			if ((cfr = (nodesA.get(i).getProduction().compareTo(nodesB.get(j)
					.getProduction()))) > 0)
				j++;
			else if (cfr < 0)
				i++;
			else {
				j_old = j;
				do {
					do {
						intersect.add(new TreeNodePairs(nodesA.get(i), nodesB
								.get(j)));

						deltaMatrix.add(nodesA.get(i).getId(), nodesB.get(j)
								.getId(), NO_RESPONSE);

						j++;
					} while (j < n_b
							&& (nodesA.get(i).getProduction().equals(nodesB
									.get(j).getProduction())));
					i++;
					j_final = j;
					j = j_old;
				} while (i < n_a
						&& (nodesA.get(i).getProduction().equals(nodesB.get(j)
								.getProduction())));
				j = j_final;
			}
		}

		return intersect;
	}

	@Override
	protected float kernelComputation(TreeRepresentation repA,
			TreeRepresentation repB) {
		return (float) evaluateKernelNotNormalize((TreeRepresentation) repA,
				(TreeRepresentation) repB);
	}

	@JsonIgnore
	public DeltaMatrix getDeltaMatrix() {
		return deltaMatrix;
	}

	@JsonIgnore
	public void setDeltaMatrix(DeltaMatrix deltaMatrix) {
		this.deltaMatrix = deltaMatrix;
	}

}
