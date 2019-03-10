package layers;

import active.ActiveFunction;
import net.Net;
import utils.BlobUtil;
import utils.MathFunctionUtil;

public class FullConnectionLayer extends Layer {
	private Net net;
	private int layerId;
	/*
	 * 层类型
	 */
	private static final String layerType = "fullconnection";
	private Blob dataAndDiff;
	/*
	 * 假设输入有50(特征map数量)*4(特征map大小)*4(特征map大小)个神经元结点， 输出有500个神经元结点，
	 * 则一共需要50*4*4*500=400000个权值参数W和500个偏置参数b 输入神经元个数
	 */
	private int inNerveCellNums;
	/*
	 * 输出神经元个数
	 */
	private int outNerveCellNums;
	/*
	 * 激活函数
	 */
	private ActiveFunction activeFunction;
	/*
	 * 权重和偏置在开始时要随机初始化，这些参数是要在训练过程中学习的,weight中保存data和diff(diff即Gradient)
	 */
	private Blob weight;
	/*
	 * 权重和偏置在开始时要随机初始化，这些参数是要在训练过程中学习的,weight中保存data和diff(diff即Gradient)
	 */
	private Blob bias;
	/*
	 * forward中产生的临时中间值，backword时需要用到，用于临时保存前面的计算结果，用于激活函数的输入
	 */
	private Blob preActiveOutput;

	public FullConnectionLayer(Net net, int inNerveCellNums, int outNerveCellNums) {
		this.net = net;
		this.inNerveCellNums = inNerveCellNums;
		this.outNerveCellNums = outNerveCellNums;
	}

	@Override
	public void prepare() {
		if (weight == null && bias == null) {
			weight = new Blob(2, inNerveCellNums, outNerveCellNums);
			weight.setData(new float[inNerveCellNums * outNerveCellNums]);
			bias = new Blob(1, outNerveCellNums);
			bias.setData(new float[outNerveCellNums]);
			// 高斯分布初始化w
			MathFunctionUtil.gaussianInitData(weight.getData());
			// 常量初始化b
			MathFunctionUtil.constantInitData(bias.getData(), 0.001f);
		}
		assert weight != null && bias != null : "FullConnectionLayer prepare---weight or bias is null error";
		weight.setDiff(new float[inNerveCellNums * outNerveCellNums]);
		bias.setDiff(new float[outNerveCellNums]);
		// preActiveOutput中间值，计算的时候要用到。
		preActiveOutput = new Blob(2, net.getBatchSize(), outNerveCellNums);
		preActiveOutput.setData(new float[net.getBatchSize() * outNerveCellNums]);
	}

	@Override
	public void initOutputDataAndDiff() {
		this.dataAndDiff = new Blob(2, net.getBatchSize(), outNerveCellNums);
		this.dataAndDiff.setData(new float[net.getBatchSize() * outNerveCellNums]);
		this.dataAndDiff.setDiff(new float[net.getBatchSize() * outNerveCellNums]);

	}

	public ActiveFunction getActiveFunction() {
		return activeFunction;
	}

	public void setActiveFunction(ActiveFunction activeFunction) {
		this.activeFunction = activeFunction;
	}

	@Override
	public void forward(Blob preLayerDataBlob) {
		// TODO Auto-generated method stub
		Blob input = preLayerDataBlob;
		Blob output = this.dataAndDiff;
		float[] inputData = input.getData();
		float[] outputData = output.getData();
		float[] wData = this.weight.getData();
		float[] bData = bias.getData();
		float[] zData = this.preActiveOutput.getData();
		BlobUtil.fillValue(zData, 0);
		for (int n = 0; n < net.getBatchSize(); n++) {
			for (int os = 0; os < this.outNerveCellNums; os++) {// 有多少个输出，当前层就有多少个神经元
				// 和每个神经元的权重相乘
				for (int is = 0; is < this.inNerveCellNums; is++) {
					// zData[n*output.get3DSize()+os] 表示一个批次中的第n个的第os个神经元
					zData[n * outNerveCellNums + os] += inputData[n * inNerveCellNums + is]
							* wData[os * inNerveCellNums + is];
				}
				// 偏执
				zData[n * outNerveCellNums + os] += bData[os];
				// 激活函数
				if (activeFunction != null) {
					outputData[n * outNerveCellNums + os] = activeFunction.dataActive(zData[n * outNerveCellNums + os]);
				} else {
					outputData[n * outNerveCellNums + os] = zData[n * outNerveCellNums + os];
				}
			}
		}
	}

	@Override
	public void backward(Blob preLayerDataAndDiffBlob) {
		// TODO Auto-generated method stub
		Blob inputDiff = this.dataAndDiff;
		Blob outputDiff = preLayerDataAndDiffBlob;
		Blob input = preLayerDataAndDiffBlob;
		float[] inputData = input.getData();
		float[] inputDiffData = inputDiff.getDiff();
		float[] outputDiffData = outputDiff.getDiff();
		float[] wData = weight.getData();
		float[] wGradientData = weight.getDiff();
		float[] bGradientData = bias.getDiff();
		float[] zData = this.preActiveOutput.getData();

		// update diff
		// 先乘激活函数的偏导数,即可求出当前层的误差
		assert inputDiff.getSize() == this.preActiveOutput.getSize() : "inputDiff.getSize()==z.getSize() error";
		if (activeFunction != null) {
			for (int n = 0; n < net.getBatchSize(); n++) {
				for (int ids = 0; ids < this.outNerveCellNums; ids++) {
					inputDiffData[n * outNerveCellNums + ids] *= activeFunction
							.diffActive(zData[n * outNerveCellNums + ids]);
				}
			}
		}
		BlobUtil.fillValue(weight.getDiff(), 0);
		for (int n = 0; n < net.getBatchSize(); n++) {
			for (int ids = 0; ids < this.outNerveCellNums; ids++) {
				for (int is = 0; is < this.inNerveCellNums; is++) {
					// 相当于一个神经元和它的每一个连接乘加
					wGradientData[ids * inNerveCellNums + is] += inputData[n * inNerveCellNums + is]
							* inputDiffData[n * outNerveCellNums + ids];
				}
			}
		}
		// 平均
		MathFunctionUtil.dataDivConstant(wGradientData, net.getBatchSize());

		// update bias
		BlobUtil.fillValue(bias.getDiff(), 0);
		for (int n = 0; n < net.getBatchSize(); n++) {
			for (int bs = 0; bs < this.outNerveCellNums; bs++) {
				bGradientData[bs] += inputDiffData[n * outNerveCellNums + bs];
			}
		}

		// 平均
		MathFunctionUtil.dataDivConstant(bGradientData, net.getBatchSize());

		// 最后，乘以当前层的权重后输出
		// 每一个输出=每一个神经元与连接他的权重的乘加
		if (this.layerId <= 2)
			return;
		BlobUtil.fillValue(outputDiff.getDiff(), 0);
		// workers.clear();
		for (int n = 0; n < net.getBatchSize(); n++) {
			for (int ids = 0; ids < this.outNerveCellNums; ids++) {
				for (int ods = 0; ods < this.inNerveCellNums; ods++) {
					outputDiffData[n * inNerveCellNums + ods] += inputDiffData[n * this.outNerveCellNums + ids]
							* wData[ids * inNerveCellNums + ods];
				}
			}
		}

		net.updateWeight(weight);
		net.updateWeight(bias);
	}

	public Blob getDataAndDiff() {
		return dataAndDiff;
	}

	public void setDataAndDiff(Blob dataAndDiff) {
		this.dataAndDiff = dataAndDiff;
	}

	public int getLayerId() {
		return layerId;
	}

	public void setLayerId(int layerId) {
		this.layerId = layerId;
	}

}
