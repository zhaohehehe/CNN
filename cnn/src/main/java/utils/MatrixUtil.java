package utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import dataset.DataSetElement;
import dataset.utils.DataSetUtil;

public class MatrixUtil {
	public static void main(String[] args) throws Exception {
		byte[] arr = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 4 };
		MatrixUtil.array2Matrix(arr, 2, 5);

		String labelPath = "data/mnist/train-labels.idx1-ubyte";
		String objPath = "data/mnist/train-images.idx3-ubyte";
		List<DataSetElement> dataSet = DataSetUtil.readMnistDataSetUtil(labelPath, objPath);
		// 测试部分图片即可
		for (int i = 0; i < dataSet.size() && i < 20; i++) {
			DataSetElement element = dataSet.get(i);
			MatrixUtil.matrix2Image(MatrixUtil.array2Matrix(element.objData, DataSetUtil.ROWS, DataSetUtil.COLUMNS),
					"D:\\zhaohe\\workspace\\CNN\\mnist\\", "" + i + "_" + element.label + ".png");
		}
	}

	public static void matrix2Image(byte[][] matrix, String saveImagePath, String saveImageName) throws IOException {
		File directory = new File(saveImagePath);
		if (!directory.exists()) {
			directory.mkdirs();
		}
		File file = new File(saveImagePath + saveImageName);
		if (!file.exists()) {
			file.createNewFile();
		}
		int rows = matrix.length;
		int colums = matrix[0].length;
		int height = rows;
		int width = colums;
		OutputStream output = new FileOutputStream(file);
		BufferedImage bufImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < colums; j++) {
				// 数组：[0,1,2,3,4,5]
				// matrix行列式：
				// |0,2,4|
				// |1,3,5|
				if (matrix[j][i] != 0) {
					bufImg.setRGB(i, j, matrix[j][i]);
				}
			}
		}

		ImageIO.write(bufImg, "png", output);

	}

	public static byte[][] array2Matrix(byte[] arr, int rows, int colums) throws Exception {
		{
			if (arr.length > rows * colums) {
				throw new Exception("目标矩阵rows*colums大于数组长度，转换失败！");
			}
			if (arr.length < rows * colums) {
				throw new Exception("目标矩阵rows*colums小于数组长度，转换失败！");
			}
			byte[][] matrix = new byte[rows][colums];
			for (int row = 0; row < rows; row++) {
				matrix[row] = Arrays.copyOfRange(arr, row * colums, (row + 1) * colums);

			}
			// 打印图片像素信息
			for (int i = 0; i < matrix.length; i++) {
				for (int j = 0; j < matrix[i].length; j++) {
					System.out.print(matrix[i][j] + " ");
				}
				System.out.println();
			}
			return matrix;
		}

	}

}
