package utils;

public class BlobUtil {

	public static void main(String[] args) {
		float[] a = new float[] { 1, 2 };
		BlobUtil.fillValue(a, 3);
		System.out.println(a[1]);

	}

	public static void fillValue(float[] data, float value) {
		for (int i = 0; i < data.length; i++) {
			data[i] = value;
		}
	}
}
