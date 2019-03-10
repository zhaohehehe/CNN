package optimizer;

import layers.Blob;

/*
 * SGD without momentum
 * https://www.jianshu.com/p/c7e642877b0e
 */

public class SGDOptimizer extends Optimizer {

	public SGDOptimizer(float lr) {
		super(lr);
	}

	public SGDOptimizer(float lr, Optimizer.GMode mode, float lamda) {
		super(lr, mode, lamda);
	}

	@Override
	public void updateBias(Blob bias) {
		// TODO Auto-generated method stub
		float[] bData = bias.getData();
		float[] gradData = bias.getDiff();
		for (int j = 0; j < bias.getSize(); j++) {
			bData[j] -= ALPHA * gradData[j];
		}
	}

	@Override
	public void updateWeight(Blob weight) {
		// TODO Auto-generated method stub
		float[] wData = weight.getData();
		float[] gradData = weight.getDiff();
		if (mode == GMode.L2) {
			for (int j = 0; j < weight.getSize(); j++) {
				wData[j] = (float) ((1 - ALPHA * LAMBDA) * wData[j] - ALPHA * gradData[j]);
			}
		} else if (mode == GMode.L1) {
			for (int j = 0; j < weight.getSize(); j++) {
				if (wData[j] >= 0) {
					wData[j] = (float) (wData[j] - ALPHA * LAMBDA - ALPHA * gradData[j]);
				} else {
					wData[j] = (float) (wData[j] + ALPHA * LAMBDA - ALPHA * gradData[j]);
				}
			}
		} else {
			for (int j = 0; j < weight.getSize(); j++) {
				wData[j] -= ALPHA * gradData[j];
			}
		}
	}
}
