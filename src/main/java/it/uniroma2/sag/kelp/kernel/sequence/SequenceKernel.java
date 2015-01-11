package it.uniroma2.sag.kelp.kernel.sequence;

import it.uniroma2.sag.kelp.data.representation.sequence.SequenceElement;
import it.uniroma2.sag.kelp.data.representation.sequence.SequenceRepresentation;
import it.uniroma2.sag.kelp.kernel.DirectKernel;

import java.util.List;

public class SequenceKernel extends DirectKernel<SequenceRepresentation> {

	// maximum length of common subsequences
	int maxSubseqLeng = 4;
	// gap penalty
	double lambda = 0.75;

	public SequenceKernel() {
		super();
	}

	public SequenceKernel(String representationIdentifier, int maxSubseqLeng,
			float lambda) {
		super(representationIdentifier);
		this.maxSubseqLeng = maxSubseqLeng;
		this.lambda = lambda;
	}

	/**
	 * Computes the number of common subsequences between two sequences.
	 * 
	 * @param s
	 *            first sequence of features.
	 * @param t
	 *            second sequence of features.
	 * @param n
	 *            maximum subsequence length.
	 * @param lambda
	 *            gap penalty.
	 * @return kernel value K[], one position for every length up to n.
	 * 
	 *         The algorithm corresponds to the recursive computation from
	 *         Figure 1 in the paper
	 *         "Subsequence Kernels for Relation Extraction" (NIPS 2005), where:
	 *         - K stands for K; - Kp stands for K'; - Kpp stands for K''; -
	 *         common stands for c;
	 */
	protected double[] stringKernel(List<SequenceElement> s,
			List<SequenceElement> t, int n, double lambda) {
		int sl = s.size();
		int tl = t.size();

		double[][][] Kp = new double[n + 1][sl][tl];

		for (int j = 0; j < sl; j++)
			for (int k = 0; k < tl; k++)
				Kp[0][j][k] = 1;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < sl - 1; j++) {
				double Kpp = 0.0;
				for (int k = 0; k < tl - 1; k++) {
					Kpp = lambda
							* (Kpp + lambda
									* elementSimilarity(s.get(j), t.get(k))
									* Kp[i][j][k]);
					Kp[i + 1][j + 1][k + 1] = lambda * Kp[i + 1][j][k + 1]
							+ Kpp;
				}
			}
		}

		double[] K = new double[n];
		for (int l = 0; l < K.length; l++) {
			K[l] = 0.0;
			for (int j = 0; j < sl; j++) {
				for (int k = 0; k < tl; k++)
					K[l] += lambda * lambda
							* elementSimilarity(s.get(j), t.get(k))
							* Kp[l][j][k];
			}
		}

		return K;
	}

	/**
	 * Computes the number of common features between two sets of featurses.
	 * 
	 * @param s
	 *            first set of features.
	 * @param t
	 *            second set of features.
	 * @return number of common features.
	 * 
	 *         The use of FeatureDictionary ensures that identical features
	 *         correspond to the same object reference. Hence, the operator '=='
	 *         can be used to speed-up the computation.
	 */
	private double elementSimilarity(SequenceElement sequenceElement,
			SequenceElement sequenceElement2) {
		if (sequenceElement.getContent().getTextFromData()
				.equals(sequenceElement2.getContent().getTextFromData()))
			return 1;
		return 0;
	}

	@Override
	protected float kernelComputation(SequenceRepresentation repA,
			SequenceRepresentation repB) {

		double[] sk = stringKernel(repA.getElements(), repB.getElements(),
				maxSubseqLeng, lambda);
		float result = 0;
		for (int i = 0; i < sk.length; i++)
			result += sk[i];

		return result;
	}

}
