package layers;

public class FullConnectionLayer {
	/*
	 * ���²���ʵ��������ʱ����
	 */
	// ����������50(����map����)*4(����map��С)*4(����map��С)����Ԫ��㣬 �����500����㣬
	// ��һ����Ҫ50*4*4*500=400000��Ȩֵ����W��500��ƫ�ò���b
	// ������Ԫ����
	private int inNerveNums;
	// �����Ԫ����
	private int outNerveNums;
	// �����
	private Object activeFunction;
}
