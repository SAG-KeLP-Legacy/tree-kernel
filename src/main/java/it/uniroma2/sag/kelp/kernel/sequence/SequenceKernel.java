package it.uniroma2.sag.kelp.kernel.sequence;

import it.uniroma2.sag.kelp.data.representation.sequence.SequenceElement;
import it.uniroma2.sag.kelp.data.representation.sequence.SequenceRepresentation;
import it.uniroma2.sag.kelp.kernel.DirectKernel;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * Sequence Kernel implementation. <br>
 * A Sequence Kernel is a convolution kernel between seqnences.The algorithm
 * corresponds to the recursive computation presented in [Bunescu&Mooney,2005]. <br>
 * <br>
 * More information at: <br>
 * [Bunescu&Mooney,2005] Razvan Bunescu and Raymond Mooney. Subsequence kernels
 * for relation extraction. In Y. Weiss, B. Scholkopf, and J. Platt, editors,
 * Advances in Neural Information Processing Systems 18, pages 171-178. MIT
 * Press, Cambridge, MA, 2006.
 * 
 * @author Danilo Croce
 * 
 */
@JsonTypeName("seqk")
public class SequenceKernel extends DirectKernel<SequenceRepresentation> {

	/**
	 * Maximum length of common subsequences
	 */
	int maxSubseqLeng = 4;
	/**
	 * Gap penalty
	 */
	double lambda = 0.75;

	public SequenceKernel() {
		super();
	}

	/**
	 * @param representationIdentifier
	 *            Identifier of the Tree representation on which the kernel
	 *            works
	 * @param maxSubseqLeng
	 *            Maximum length of common subsequences
	 * @param lambda
	 *            Gap penalty
	 */
	public SequenceKernel(String representationIdentifier, int maxSubseqLeng,
			float lambda) {
		super(representationIdentifier);
		this.maxSubseqLeng = maxSubseqLeng;
		this.lambda = lambda;
	}

	/**
	 * Computes a simple the Identity similarity function between two element of
	 * the sequence.
	 * 
	 * @param se1
	 *            an element from the first sequence
	 * 
	 * @param se2
	 *            an element from the second sequence
	 * 
	 * @return 1 if both elements have the same type and label. 0 otherwise
	 */
	private double elementSimilarity(SequenceElement se1, SequenceElement se2) {
		if (!se1.getContent().getClass().equals(se2.getContent().getClass()))
			return 0;

		if (se1.getContent().getTextFromData()
				.equals(se2.getContent().getTextFromData()))
			return 1;
		return 0;
	}

	/**
	 * @return Gap penalty
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * @return Maximum length of common subsequences
	 */
	public int getMaxSubseqLeng() {
		return maxSubseqLeng;
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

	/**
	 * @param lambda
	 *            Gap penalty
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * @param maxSubseqLeng
	 *            Maximum length of common subsequences
	 */
	public void setMaxSubseqLeng(int maxSubseqLeng) {
		this.maxSubseqLeng = maxSubseqLeng;
	}

	/**
	 * Computes the number of common subsequences between two sequences. The
	 * algorithm corresponds to the recursive computation from Figure 1 in the
	 * paper [Bunescu&Mooney,2005] where: - K stands for K; - Kp stands for K';
	 * - Kpp stands for K''; - common stands for c;
	 * 
	 * @param s
	 *            first sequence
	 * @param t
	 *            second sequence
	 * @param n
	 *            maximum subsequence length.
	 * @param lambda
	 *            gap penalty
	 * 
	 * @return kernel value K[], one position for every length up to n.
	 * 
	 * 
	 */
	private double[] stringKernel(List<SequenceElement> s,
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

}
