package dataset;

public class ZScore {
	//矩阵归一化
	public static double[][] Normalization(double[][] matrix) {
		int rows=matrix.length;
		int cols=matrix[0].length;
		double[][] result=new double[rows][cols];
		for(int i=0;i<rows;i++){
			double mean=getMean(matrix[i]);
			double sdev=getStandardDev(matrix[i]);
			for(int j=0;j<cols;j++){
				result[i][j]=(matrix[i][j]-mean)/sdev;
			}
		}
		return result;
	}
	
	//获取平均值
	public static double getMean(double[] entries) {
		double out = 0;
		for(double d: entries) {
			out += d/entries.length;
		}		
		return out;
	}
	//标准差
	public static double getStandardDev(double[] entries){
		double mean=getMean(entries);
		double sum=0;
		for(int i=0;i<entries.length;i++){
			sum+=Math.sqrt((entries[i]-mean)*(entries[i]-mean));
		}
		return (sum/(entries.length-1));	
	}

	public static void main(String[] args) {
		// TODO 自动生成的方法存根
		double[][] db=new double[][]{{95,85,75,65,55,45}};
		double[][] result = ZScore.Normalization(db);
		for(int i=0;i<result[0].length;i++){
			System.out.println(result[0][i]+" ");
		}
		
		byte[][] test=new byte[][]{{127,85,75,65,55,45}};
		for(int i=0;i<test[0].length;i++){
			//byte[-127,128]不能直接转int,然后除以128 
			System.out.println((test[0][i] & 0xff) / 128.0d - 1+" ");
			System.out.println((test[0][i] & 0xff) / 128.0f - 1+" ");
		}
		
		
		
		

	}

}
