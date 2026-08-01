package constraint;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

import ftchecker.Tuple;
import engine.Main;

public class ForbiddenTuples {

	 public static int[] getDomains(String file)
	  {
	    try {
	      Scanner s = new Scanner(new File(file));

	      int n = s.nextInt();
	      Main.paraNum = n;
	      int[] domains = new int[n];

	      for (int i = 0; i < n; i++) {
	        domains[i] = s.nextInt();
	      }

	      s.close();
	      return domains;
	    }
	    catch (FileNotFoundException e)
	    {
	      e.printStackTrace();
	    }
	    return null;
	  }

	 public static ArrayList<Integer> getIndexToParamMap(int[] domains) {
	    ArrayList <Integer> list = new ArrayList<Integer>();
	    for(int i=0; i<domains.length; i++){
	    	for(int v=0; v<domains[i]; v++)
	    		list.add(Integer.valueOf(i));
	    }
	    return list;

	  }

	  public static ArrayList<Integer> getIndexToValueMap(int[] domains) {
	    ArrayList<Integer> list = new ArrayList<Integer>();
	    for(int i=0; i<domains.length; i++){
	    	for(int v=0; v<domains[i]; v++)
	    		list.add(Integer.valueOf(v));
	    }
	    return list;
	  }

	  public  static ArrayList<Tuple> getForbiddenTuples(String file, ArrayList<Integer> indexToParamMap, ArrayList<Integer> indexToValueMap)
	  {
	    try
	    {
	      Scanner s = new Scanner(new File(file));
	      int n = s.nextInt();
	      ArrayList<Tuple> fts = new ArrayList<Tuple>(n);

	      for (int i = 0; i < n; i++) {
	        int len = s.nextInt();
	        ArrayList<int []> pvs = new ArrayList<>();
	        for (int j = 0; j < len; j++) {
	          s.next();
	          int index = s.nextInt();

	          int[] pv = new int[2];
	          pv[0] = indexToParamMap.get(index);
	          pv[1] = indexToValueMap.get(index);
	          pvs.add(pv);
	        }
	        s.nextLine();

	        Collections.sort(pvs, new Comparator<int[]>()
	        {
	          public int compare(int[] arg0, int[] arg1) {
	            if (arg0[0] == arg1[0]) {
	              System.out.println("wrong tuple " + arg0[0] + " " + arg1[0]);
	            }
	            return arg0[0] - arg1[0];
	          }
	        });
	        ArrayList<Integer> values = new ArrayList<Integer>();
	        for (int[] pv : pvs) {
	          values.add(pv[0]);
	          values.add(pv[1]);
	        }
	        fts.add(new Tuple(values));
	      }

	      s.close();
	      return fts;
	    }
	    catch (FileNotFoundException e)
	    {
	      e.printStackTrace();
	    }
	    return null;
	  }

}
