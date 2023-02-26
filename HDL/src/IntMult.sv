/*
 
  
 */

`timescale 1ns/1ns

module IntMult
  #(
    parameter integer  W_IN_A =8,
    parameter integer  W_IN_B =16,
    localparam integer W_OUT_X =W_IN_B+W_IN_A
    )
   (
    input logic [W_IN_A-1:0]   in_a,
    input logic [W_IN_B-1:0]   in_b,
    output logic [W_OUT_X-1:0] out_x
    );

   logic 		       sign_of_a;
   logic 		       sign_of_b;
   logic 		       sign_of_x;
   logic [W_IN_A-1:0] 	       in_a_unsigned;
   logic [W_IN_B-1:0] 	       in_b_unsigned;
   logic [W_OUT_X-1:0] 	       out_x_unsigned;

   assign sign_of_a		= in_a[W_IN_A-1];
   assign sign_of_b		= in_b[W_IN_B-1];
   assign sign_of_x		= sign_of_b^sign_of_a;      
   assign in_a_unsigned		= (sign_of_a) ? -in_a : in_a;   
   assign in_b_unsigned		= (sign_of_b) ? -in_b : in_b;
   assign out_x_unsigned	= in_a_unsigned * in_b_unsigned;
   assign out_x			= (sign_of_x) ? -out_x_unsigned : out_x_unsigned;      

endmodule // IntMult

