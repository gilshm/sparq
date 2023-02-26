
`timescale 1ns/1ns

module FpMultTb;
   
   localparam integer W_MANTISSA        = 10;
   localparam integer W_EXPONENT	= 5;
   localparam integer W_FP_NUMBER	= W_MANTISSA + W_EXPONENT + 1;
   localparam integer N_TESTS		= 50;

   logic 	      clk=1'b0;
   always #5 clk=~clk;

   // *****************************************
   logic [W_FP_NUMBER-1:0] in_a;
   logic [W_FP_NUMBER-1:0] in_b;
   logic [W_FP_NUMBER-1:0] out_x;
   
   logic 		   sign [1:0];
   logic [W_EXPONENT-1:0]  exponent [1:0];
   logic [W_MANTISSA-1:0]  mantissa [1:0];

   logic 		   sign_x;
   logic [W_EXPONENT-1:0]  exponent_x;
   logic [W_MANTISSA-1:0]  mantissa_x;
   
   initial begin
      for(int i=0; i<N_TESTS; i++) begin
	 @(posedge clk);
	 for(int n=0; n<2; n++) begin
	    sign[n]     = $random();
	    exponent[n] = $random();
	    mantissa[n] = $random();
	 end
      end
   end
   
   assign in_a = {sign[0], exponent[0], mantissa[0]};
   assign in_b = {sign[1], exponent[1], mantissa[1]};
   assign sign_x     = out_x[W_FP_NUMBER-1];
   assign exponent_x = out_x[W_MANTISSA+:W_EXPONENT];
   assign mantissa_x = out_x[W_MANTISSA-1:0]; 
   
   // *****************************************    
   int 			   exp_val [2:0];

   assign exp_val[0] = (exponent[0]-(2**(W_EXPONENT-1)-1));
   assign exp_val[1] = (exponent[1]-(2**(W_EXPONENT-1)-1));
   assign exp_val[2] = (exponent_x-(2**(W_EXPONENT-1)-1));

   // *****************************************    
   shortreal 		   mantissa_val [2:0];

   always_comb begin
      mantissa_val[0] = '0;	
      if(exponent[0]!='0) mantissa_val[0] += 1.0;
      for(int i=0; i<W_MANTISSA; i=i+1) begin
	 if(mantissa[0][W_MANTISSA-1-i]) mantissa_val[0] += 0.5**(i+1); 
      end
   end

   always_comb begin
      mantissa_val[1] = '0;	
      if(exponent[1]!='0) mantissa_val[1] += 1.0;
      for(int i=0; i<W_MANTISSA; i=i+1) begin
	 if(mantissa[1][W_MANTISSA-1-i]) mantissa_val[1] += 0.5**(i+1); 
      end
   end
   
   always_comb begin
      mantissa_val[2] = '0;	
      if(exponent_x!='0) mantissa_val[2] += 1.0;
      for(int i=0; i<W_MANTISSA; i=i+1) begin
	 if(mantissa_x[W_MANTISSA-1-i]) mantissa_val[2] += 0.5**(i+1); 
      end
   end
   
   // *****************************************
   shortreal 		   float_a;
   shortreal  		   float_b;
   shortreal 		   float_x;
   shortreal 		   expected_x;
   logic unsigned [31:0]   conv2float;

   assign float_a = (sign[0] ? -1.0 : 1.0)*(2.0**(exp_val[0]))*mantissa_val[0];
   assign float_b = (sign[1] ? -1.0 : 1.0)*(2.0**(exp_val[1]))*mantissa_val[1];
   assign float_x = (sign[2] ? -1.0 : 1.0)*(2.0**(exp_val[2]))*mantissa_val[2];
   assign expected_x = float_a*float_b;
   
   assign conv2float[31]    = sign_x;
   assign conv2float[30:23] = exponent_x - (2**(W_EXPONENT-1)-1) + {1'b0, {7{1'b1}}};
   assign conv2float[22:0]  = {mantissa_x, {23-W_MANTISSA{1'b0}}};

   
   // *****************************************
   initial begin
      $display("%s",{30{"#"}});
      $display("Starting Fp test");
      $display("Width of Exponent          = %0d-bits", W_EXPONENT);
      $display("Width of Mantissa          = %0d-bits", W_MANTISSA);
      $display("Width of Fp representation = %0d-bits", W_FP_NUMBER);
      $display("%s",{30{"#"}});
      for(int i=0; i<N_TESTS; i++) begin
	 @(negedge clk);
	 $display("%s",{90{"*"}});
	 //$write("Raw A=%16b: S=%1b, E=%5b, M=%10b #", in_a, sign[0], exponent[0], mantissa[0]);
	 //$write("Float A: Exp val=%0d, mantissa val=%f, float val=%f\n", exp_val[0], mantissa_val[0], float_a);
	 //$write("Raw B=%16b: S=%1b, E=%5b, M=%10b #", in_b, sign[1], exponent[1], mantissa[1]);
	 //$write("Float B: Exp val=%0d, mantissa val=%f, float val=%f\n", exp_val[1], mantissa_val[1], float_b);
	 //$write("Raw X=%16b: S=%1b, E=%5b, M=%10b #", out_x, sign_x, exponent_x, mantissa_x);
	 //$write("Float X: Exp val=%0d, mantissa val=%f, float val=%f, expected=%f\n", exp_val[2], mantissa_val[2], float_x, expected_x);	 
	 $write("Float X: float val=%f, expected=%f\n", float_x, expected_x);	 
	 //$write("Results  X: Binary=%32b, float=%f, expected=%f\n", conv2float, shortreal'(conv2float), expected_x);	 
      end
      $display("%s",{90{"#"}});
      $finish();
   end

   FpMult
     #(.W_MANTISSA(W_MANTISSA),
       .W_EXPONENT(W_EXPONENT))
   DUT
     (
      .in_a		(in_a),
      .in_b		(in_b),
      .out_x		(out_x),
      .overflow		(),
      .underflow	(),
      .exception        ()
      );
   
endmodule // IntMultTwosComplimentTb
