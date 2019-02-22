package layers;

public class ConvolutionLayer {
	// 层类型
	private String layerType = "convolution";
	// 卷积核(权重)，卷积核和偏置在开始时要随机初始化，这些参数是要在训练过程中学习的
	private Object kernel;
	// 偏置矩阵，卷积核和偏置在开始时要随机初始化，这些参数是要在训练过程中学习的
	private Object bias;

	private Object kernelGradient;
	private Object biasGradient;

	
	/*
	 * 以下参数实例化对象时给出
	 */
	// 输入特征Map宽度
	private int width;
	// 输入特征Map高度
	private int height;
	// 输入特征Map数
	private int inMapNums;
	// 输出特征Map数
	private int outMapNums;
	// 卷积核大小
	private int kernelSize;
	// 卷积扫描步长
	private int stride;
	//激活函数
	private Object activeFunction;
	

}
