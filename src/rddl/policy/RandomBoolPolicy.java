/**
 * RDDL: Implements a fixed policy for a domain with a single boolean action.
 *       The action selected to be true is the first ground instance.
 * 
 * @author Scott Sanner (ssanner@gmail.com)
 * @version 10/10/10
 *
 **/

package rddl.policy;

import java.util.*;

import rddl.*;
import rddl.RDDL.*;

public class RandomBoolPolicy extends Policy {
	
	public RandomBoolPolicy(String instance_name) {
		super(instance_name);
	}

	public ArrayList<PVAR_INST_DEF> getActions(State s) throws EvalException {
		
		// Get a random action
		PVAR_NAME p = s._alActionNames.get(0);
		
		// Get term instantations for that action and select *one*
		ArrayList<ArrayList<LCONST>> inst = s.generateAtoms(p);
		ArrayList<LCONST> terms = inst.get(new Random().nextInt(inst.size()));
		
		// Generate the action list
		PVAR_INST_DEF d = new PVAR_INST_DEF(p._sPVarName, new Boolean(true), terms);
		ArrayList<PVAR_INST_DEF> actions = new ArrayList<PVAR_INST_DEF>();
		actions.add(d);
		return actions;
	}

}
