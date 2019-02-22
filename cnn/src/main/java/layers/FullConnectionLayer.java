package layers;

public class FullConnectionLayer {
	/*
	 * 以下参数实例化对象时给出
	 */
	// 假设输入有50(特征map数量)*4(特征map大小)*4(特征map大小)个神经元结点， 输出有500个结点，
	// 则一共需要50*4*4*500=400000个权值参数W和500个偏置参数b
	// 输入神经元个数
	private int inNerveNums;
	// 输出神经元个数
	private int outNerveNums;
	// 激活函数
	private Object activeFunction;
}
