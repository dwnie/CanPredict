package parallel;

import java.io.Serializable;

import engine.Main;
import struct.Element;

public class NeighborEvalSerial1 implements Serializable,
		NeighborEval {
	private static final long serialVersionUID = 1L;

	private int row;

	private int column;

	private int newvalue;

	private Element[] neighbors;

	private int no;

	public NeighborEvalSerial1(int row, int column, int newvalue,  Element[] neighbors,int no){
		this.row = row;
		this.column = column;
		this.newvalue = newvalue;
		this.neighbors = neighbors ;
		this.no = no;
	}

	public void caculEffcAndTabu(){
		int effect1,effect2;

		int[] test = new int[Main.paraNum*2];
		for(int j=0; j<Main.paraNum; j++ )
		{
			test[2*j] = j;
			test[2*j+1] = Main.coverageArray[row][j];
		}

		if(Main.numMFTParam[column]>0)
			effect1 = Main.checker.evalMove_ParaMFS(test, column, Main.coverageArray[row][column], newvalue);

		else
			effect1 = 0;

		effect2 = Main.evalMove.evaluate_fast(row,column,newvalue);

		boolean istabu = Main.tabuList.isTabu(row,column,newvalue);

		neighbors[no] = new Element(row,column,newvalue,effect1,effect2,istabu);

	}

}
