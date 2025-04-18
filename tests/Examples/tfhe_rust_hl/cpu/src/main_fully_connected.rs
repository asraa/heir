#[cfg(test)]
mod test {
    use fully_connected_test_rs_lib;

    use tfhe::{ConfigBuilder, generate_keys, set_server_key, FheUint8};
    use tfhe::prelude::*;

    #[test]
    fn simple_test() {
      // Input = 2, output = 2 * 2 + 1 = 5
      let config = ConfigBuilder::default().build();

      // Client-side
      let (client_key, server_key) = generate_keys(config);

      let a: tfhe::FheUint<tfhe::FheUint8Id> = FheUint8::encrypt(2u8, &client_key);
      let input_vec = core::array::from_fn(|_1| core::array::from_fn(|_1| a.clone()));

      set_server_key(server_key);

      let result = fully_connected_test_rs_lib::fn_under_test(&input_vec);

      let output: u32 = result[0][0].decrypt(&client_key);
      assert_eq!(output, 5);
    }
}
