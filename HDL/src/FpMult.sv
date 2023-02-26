
`timescale 1ns/1ns

module FpMult
  #(
    parameter integer  W_MANTISSA = 8,
    parameter integer  W_EXPONENT = 8,
    localparam integer W_FP_NUMBER = W_MANTISSA + W_EXPONENT + 1 // 1 for sign bit 
    )
   (
    input logic [W_FP_NUMBER-1:0]  in_a,
    input logic [W_FP_NUMBER-1:0]  in_b,
    output logic [W_FP_NUMBER-1:0] out_x,
    output logic 		   overflow,
    output logic 		   underflow,
    output logic 		   exception
    );

   logic 			   round;
      
   // sign
   logic 			   sign_of_a;
   logic 			   sign_of_b;
   logic 			   sign_of_x;
   // exponent 
   logic [W_EXPONENT-1:0] 	   exponent_of_a;
   logic [W_EXPONENT-1:0] 	   exponent_of_b;
   logic [W_EXPONENT-1:0] 	   exponent_of_x;
   logic [W_EXPONENT:0] 	   exponent_of_x_result;
   logic 			   exponent_of_a_is_zero;
   logic 			   exponent_of_b_is_zero;
   logic [W_EXPONENT-1:0] 	   exponent_bias;
   // mantissa
   logic [W_MANTISSA-1:0] 	   mantissa_of_a;
   logic [W_MANTISSA-1:0] 	   mantissa_of_b;
   logic [W_MANTISSA-1:0] 	   mantissa_of_x;
   logic [W_MANTISSA:0] 	   mantissa_of_a_extended;
   logic [W_MANTISSA:0] 	   mantissa_of_b_extended;
   logic 			   mantissa_mult_overflow;
   logic [(2*(W_MANTISSA+1))-1:0]  mantissa_mult_a_b;
   logic [(2*(W_MANTISSA+1))-1:0]  mantissa_mult_a_b_shifted;
   
   // sign calculaions
   assign sign_of_a		= in_a[W_FP_NUMBER-1];
   assign sign_of_b		= in_b[W_FP_NUMBER-1];
   assign sign_of_x		= sign_of_a ^ sign_of_b;
   
   // exponent calculations
   assign exponent_bias         = {1'b0, {(W_EXPONENT-1){1'b1}}};
   assign exponent_of_a		= in_a[W_MANTISSA+:W_EXPONENT];
   assign exponent_of_b		= in_b[W_MANTISSA+:W_EXPONENT];
   assign exponent_of_a_is_zero = (exponent_of_a=='0);
   assign exponent_of_b_is_zero = (exponent_of_b=='0);
   assign exception             = (exponent_of_a=='1) | (exponent_of_b=='1);
   
   // Full-Adder + Full-Subtractor
   assign exponent_of_x = exponent_of_x_result[W_EXPONENT-1:0];
   always_comb begin
      exponent_of_x_result  = (exponent_of_a + exponent_of_b) - exponent_bias;   
      if(mantissa_mult_overflow) 
	exponent_of_x_result += 1;
   end
   
   // mantissa calculations
   assign mantissa_of_a		 = in_a[W_MANTISSA-1:0];
   assign mantissa_of_b		 = in_b[W_MANTISSA-1:0];
   assign mantissa_of_a_extended = {~exponent_of_a_is_zero, mantissa_of_a}; 
   assign mantissa_of_b_extended = {~exponent_of_b_is_zero, mantissa_of_b}; 
   assign mantissa_mult_a_b      = mantissa_of_a_extended * mantissa_of_b_extended;

   // Unsigned Multiplier W_MANTISSA x W_MANTISSA
   assign mantissa_mult_overflow    = mantissa_mult_a_b[(2*(W_MANTISSA+1))-1];
   assign mantissa_mult_a_b_shifted = (mantissa_mult_overflow ? mantissa_mult_a_b : mantissa_mult_a_b<<1);
   assign round                     = |mantissa_mult_a_b_shifted[W_MANTISSA-1:0];
   assign zero                      = (mantissa_of_x=='0) & (~exception);
   
   always_comb begin
      mantissa_of_x = mantissa_mult_a_b_shifted[(W_MANTISSA+1)+:W_MANTISSA];
      if(round & mantissa_mult_a_b_shifted[W_MANTISSA])
	mantissa_of_x += 1;      
   end

   assign overflow  = (~zero & (exponent_of_x_result[W_EXPONENT] & ~exponent_of_x_result[W_EXPONENT-1]));
   assign underflow = (~zero & (exponent_of_x_result[W_EXPONENT] & exponent_of_x_result[W_EXPONENT-1]));
   
   // x result recontstuction 
   always_comb begin
      if(exception)      out_x = '0;
      else if(zero)      out_x = {sign_of_x, {W_FP_NUMBER-1{1'b0}}};
      else if(overflow)  out_x = {sign_of_x, {W_EXPONENT{1'b1}}, {W_MANTISSA{1'b1}}};
      else if(underflow) out_x = {sign_of_x, {W_FP_NUMBER-1{1'b0}}};
      else               out_x = {sign_of_x, exponent_of_x, mantissa_of_x};
   end
   
endmodule // FpMult
