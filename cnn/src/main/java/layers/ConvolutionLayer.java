package layers;

public class ConvolutionLayer {
	// ������
	private String layerType = "convolution";
	// �����(Ȩ��)������˺�ƫ���ڿ�ʼʱҪ�����ʼ������Щ������Ҫ��ѵ��������ѧϰ��
	private Object kernel;
	// ƫ�þ��󣬾���˺�ƫ���ڿ�ʼʱҪ�����ʼ������Щ������Ҫ��ѵ��������ѧϰ��
	private Object bias;

	private Object kernelGradient;
	private Object biasGradient;

	
	/*
	 * ���²���ʵ��������ʱ����
	 */
	// ��������Map���
	private int width;
	// ��������Map�߶�
	private int height;
	// ��������Map��
	private int inMapNums;
	// �������Map��
	private int outMapNums;
	// ����˴�С
	private int kernelSize;
	// ���ɨ�貽��
	private int stride;
	//�����
	private Object activeFunction;
	

}
