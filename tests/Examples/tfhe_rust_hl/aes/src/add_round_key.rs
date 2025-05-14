use std::collections::HashMap;
use tfhe::shortint::prelude::*;

use tfhe::shortint::server_key::LookupTableOwned;

enum GateInput {
    Tv(usize), // key in a global hashmap
}

use GateInput::*;

enum OpType<'a> {
    LUT3(&'a str), // key in a global hashmap
    ADD,
    LSH(u8), // shift value
}

use OpType::*;

pub fn add_round_key(
  v0: &ServerKey,
  v1: &[Ciphertext; 16],
  v2: &[Ciphertext; 16],
) -> [Ciphertext; 16] {
  let v3 = 15;
  let v4 = 14;
  let v5 = 13;
  let v6 = 12;
  let v7 = 11;
  let v8 = 10;
  let v9 = 9;
  let v10 = 8;
  let v11 = 7;
  let v12 = 6;
  let v13 = 5;
  let v14 = 4;
  let v15 = 3;
  let v16 = 2;
  let v17 = 1;
  let v18 = 0;
  let v19 = v0.generate_lookup_table(|x| (6 >> x) & 1);
  let mut v20 : HashMap<(usize), Ciphertext> = HashMap::new();
  let v21 = &v1[v18];
  let v22 = &v2[v18];
  let v23 = v0.scalar_left_shift(&v22, 1 as u8);
  let v24 = v0.unchecked_add(&v23, &v21);
  let v25 = v0.apply_lookup_table(&v24, &v19);
  v20.insert((v18 as usize), v25);
  let v26 = &v1[v17];
  let v27 = &v2[v17];
  let v28 = v0.scalar_left_shift(&v27, 1 as u8);
  let v29 = v0.unchecked_add(&v28, &v26);
  let v30 = v0.apply_lookup_table(&v29, &v19);
  v20.insert((v17 as usize), v30);
  let v31 = &v1[v16];
  let v32 = &v2[v16];
  let v33 = v0.scalar_left_shift(&v32, 1 as u8);
  let v34 = v0.unchecked_add(&v33, &v31);
  let v35 = v0.apply_lookup_table(&v34, &v19);
  v20.insert((v16 as usize), v35);
  let v36 = &v1[v15];
  let v37 = &v2[v15];
  let v38 = v0.scalar_left_shift(&v37, 1 as u8);
  let v39 = v0.unchecked_add(&v38, &v36);
  let v40 = v0.apply_lookup_table(&v39, &v19);
  v20.insert((v15 as usize), v40);
  let v41 = &v1[v14];
  let v42 = &v2[v14];
  let v43 = v0.scalar_left_shift(&v42, 1 as u8);
  let v44 = v0.unchecked_add(&v43, &v41);
  let v45 = v0.apply_lookup_table(&v44, &v19);
  v20.insert((v14 as usize), v45);
  let v46 = &v1[v13];
  let v47 = &v2[v13];
  let v48 = v0.scalar_left_shift(&v47, 1 as u8);
  let v49 = v0.unchecked_add(&v48, &v46);
  let v50 = v0.apply_lookup_table(&v49, &v19);
  v20.insert((v13 as usize), v50);
  let v51 = &v1[v12];
  let v52 = &v2[v12];
  let v53 = v0.scalar_left_shift(&v52, 1 as u8);
  let v54 = v0.unchecked_add(&v53, &v51);
  let v55 = v0.apply_lookup_table(&v54, &v19);
  v20.insert((v12 as usize), v55);
  let v56 = &v1[v11];
  let v57 = &v2[v11];
  let v58 = v0.scalar_left_shift(&v57, 1 as u8);
  let v59 = v0.unchecked_add(&v58, &v56);
  let v60 = v0.apply_lookup_table(&v59, &v19);
  v20.insert((v11 as usize), v60);
  let v61 = &v1[v10];
  let v62 = &v2[v10];
  let v63 = v0.scalar_left_shift(&v62, 1 as u8);
  let v64 = v0.unchecked_add(&v63, &v61);
  let v65 = v0.apply_lookup_table(&v64, &v19);
  v20.insert((v10 as usize), v65);
  let v66 = &v1[v9];
  let v67 = &v2[v9];
  let v68 = v0.scalar_left_shift(&v67, 1 as u8);
  let v69 = v0.unchecked_add(&v68, &v66);
  let v70 = v0.apply_lookup_table(&v69, &v19);
  v20.insert((v9 as usize), v70);
  let v71 = &v1[v8];
  let v72 = &v2[v8];
  let v73 = v0.scalar_left_shift(&v72, 1 as u8);
  let v74 = v0.unchecked_add(&v73, &v71);
  let v75 = v0.apply_lookup_table(&v74, &v19);
  v20.insert((v8 as usize), v75);
  let v76 = &v1[v7];
  let v77 = &v2[v7];
  let v78 = v0.scalar_left_shift(&v77, 1 as u8);
  let v79 = v0.unchecked_add(&v78, &v76);
  let v80 = v0.apply_lookup_table(&v79, &v19);
  v20.insert((v7 as usize), v80);
  let v81 = &v1[v6];
  let v82 = &v2[v6];
  let v83 = v0.scalar_left_shift(&v82, 1 as u8);
  let v84 = v0.unchecked_add(&v83, &v81);
  let v85 = v0.apply_lookup_table(&v84, &v19);
  v20.insert((v6 as usize), v85);
  let v86 = &v1[v5];
  let v87 = &v2[v5];
  let v88 = v0.scalar_left_shift(&v87, 1 as u8);
  let v89 = v0.unchecked_add(&v88, &v86);
  let v90 = v0.apply_lookup_table(&v89, &v19);
  v20.insert((v5 as usize), v90);
  let v91 = &v1[v4];
  let v92 = &v2[v4];
  let v93 = v0.scalar_left_shift(&v92, 1 as u8);
  let v94 = v0.unchecked_add(&v93, &v91);
  let v95 = v0.apply_lookup_table(&v94, &v19);
  v20.insert((v4 as usize), v95);
  let v96 = &v1[v3];
  let v97 = &v2[v3];
  let v98 = v0.scalar_left_shift(&v97, 1 as u8);
  let v99 = v0.unchecked_add(&v98, &v96);
  let v100 = v0.apply_lookup_table(&v99, &v19);
  v20.insert((v3 as usize), v100);
  core::array::from_fn(|i0| v20.get(&(i0)).unwrap().clone())
}
