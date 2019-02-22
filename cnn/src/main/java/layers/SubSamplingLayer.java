package layers;

public class SubSamplingLayer {
	/*
	 * 以下参数实例化对象时给出
	 */
	// 输入特征Map宽度
	private int width;
	// 输入特征Map高度
	private int height;
	// 输入特征Map数
	private int inMapNums;
	// 卷积核大小
	private int kernelSize;
	// 卷积扫描步长
	private int stride;
	//激活函数
	private Object activeFunction;
}
