
`timescale 1ns/1ns

module IntMultTb;
   
   localparam integer  W_IN_A	=8;
   localparam integer  W_IN_B	=5;
   localparam integer  W_OUT_X	=W_IN_B+W_IN_A;
   localparam integer  N_TESTS	=10;
   
   logic signed [W_IN_A-1:0] in_a	='0;
   logic signed [W_IN_B-1:0] in_b	='0;
   logic signed [W_OUT_X-1:0] out_x	='0;

   logic 	       clk=1'b0;
   always #5 clk=~clk;
   
   initial begin
      for(int i=0; i<N_TESTS; i++) begin
	 @(posedge clk);
	 in_a = $random();
	 in_b = $random();
	 @(negedge clk);
	 $display("Calc %0d: A=%0d, B=%0d, Out=%0d", i, in_a, in_b, out_x);
	 assert(out_x == in_a*in_b);
      end
      $finish();
   end

   IntMult
     #(.W_IN_A(W_IN_A),
       .W_IN_B(W_IN_B))
   DUT
     (
      .in_a(in_a),
      .in_b(in_b),
      .out_x(out_x)
      );
   
endmodule // IntMultTwosComplimentTb
