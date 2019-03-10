package optimizer;

import layers.Blob;

public abstract class Optimizer {
	/*
	 * α在梯度下降算法中被称作为学习率或者步长，意味着我们可以通过α来控制每一步走的距离，以保证步子不要走太快，错过了最低点。同时也要保证不要走的太慢，
	 * 导致太阳下山了，还没有走到山下。所以α的选择在梯度下降法中往往是很重要的！α不能太大也不能太小，太小的话，可能导致迟迟走不到最低点，太大的话，
	 * 会导致错过最低点！ https://www.jianshu.com/p/c7e642877b0e
	 */
	protected float ALPHA = 0;
	protected float LAMBDA = 0;

	public static enum GMode {
		NONE, L1, L2
	}

	GMode mode;

	public Optimizer(float lr) {
		ALPHA = lr;
		this.mode = GMode.NONE;
	}

	public Optimizer(float lr, GMode mode, float lamda) {
		ALPHA = lr;
		LAMBDA = lamda;
		this.mode = mode;
	}
	// 优化算法的功能，是通过改善训练方式，来最小化(或最大化)损失函数Error(x)。
	// 模型内部有些参数，是用来计算测试集中目标值Y的真实值和预测值的偏差程度的，基于这些参数，就形成了损失函数Error(x)。
	// 比如说，权重(W)和偏差(b)就是这样的内部参数，一般用于计算输出值，在训练神经网络模型时起到主要作用。
	// 在有效地训练模型并产生准确结果时，模型的内部参数起到了非常重要的作用。这也是为什么我们应该用各种优化策略和算法，来更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值

	public abstract void updateWeight(Blob weight);

	public abstract void updateBias(Blob bias);

	public void setLr(float lr) {
		ALPHA = lr;
	}

	public float getLr() {
		return ALPHA;
	}

}
