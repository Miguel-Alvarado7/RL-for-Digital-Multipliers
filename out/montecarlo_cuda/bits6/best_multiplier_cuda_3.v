
`timescale 1ns/1ps
module multiplier (
    input [5:0] A,
    input [5:0] B,
    output [11:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = 0;
    wire pp1 = (A[5]&~B[4]);
    wire pp2 = (A[3]&~B[2]);
    wire pp3 = (A[0]&B[1]);
    wire pp4 = (A[2]&~B[5]);
    wire pp5 = (~A[3]&~B[1]);
    wire pp6 = (A[2]&~B[2]);
    wire pp7 = (~A[2]&~B[3]);
    wire pp8 = (~A[0]&~B[5]);
    wire pp9 = (A[4]&B[1]);
    wire pp10 = (A[2]&~B[3]);
    wire pp11 = (~A[4]&~B[0]);
    wire pp12 = (~A[5]&~B[5]);
    wire pp13 = (A[5]&~B[5]);
    wire pp14 = (A[3]&~B[5]);
    wire pp15 = (A[1]&B[4]);
    wire pp16 = (A[1]&B[5]);
    wire pp17 = (A[4]&B[4]);
    wire pp18 = (~A[1]&B[3]);
    wire pp19 = (~A[5]&B[1]);
    wire pp20 = (~A[3]&~B[3]);
    wire pp21 = (A[3]&B[3]);
    wire pp22 = (~A[5]&~B[2]);
    wire pp23 = (A[5]&B[4]);
    wire pp24 = (A[3]&B[2]);
    wire pp25 = (A[4]&B[5]);
    wire pp26 = (~A[2]&~B[5]);
    wire pp27 = (~A[2]&B[0]);
    wire pp28 = (A[2]&~B[4]);
    wire pp29 = (A[5]&~B[1]);
    wire pp30 = (~A[1]&B[4]);
    wire pp31 = (A[1]&~B[5]);
    wire pp32 = (~A[1]&B[2]);
    wire pp33 = (A[5]&B[1]);
    wire pp34 = (~A[0]&~B[3]);
    wire pp35 = (A[2]&B[1]);
    wire pp36 = (~A[0]&~B[0]);
    wire pp37 = (A[2]&B[2]);
    wire pp38 = (A[1]&~B[1]);
    wire pp39 = (~A[1]&B[0]);
    wire pp40 = (A[0]&~B[5]);
    wire pp41 = (~A[1]&~B[3]);
    wire pp42 = (A[5]&~B[3]);
    wire pp43 = (~A[2]&~B[1]);
    wire pp44 = (~A[5]&B[5]);
    wire pp45 = (A[4]&~B[0]);
    wire pp46 = (A[0]&B[2]);
    wire pp47 = (~A[4]&B[1]);

    // Suma de productos parciales
    wire [5:0] columna12 = pp0 + pp12 + pp8 + pp14 + pp40;
    wire [5:0] columna11 = pp1 + pp13 + pp23 + pp14;
    wire [5:0] columna10 = pp2 + pp14 + pp24 + pp31 + pp23;
    wire [5:0] columna9 = pp3 + pp15 + pp17 + pp32 + pp41 + pp13;
    wire [5:0] columna8 = pp4 + pp16 + pp25 + pp33 + pp14 + pp19;
    wire [5:0] columna7 = pp5 + pp17 + pp26 + pp6 + pp41 + pp46;
    wire [5:0] columna6 = pp6 + pp5 + pp17 + pp34 + pp23 + pp25;
    wire [5:0] columna5 = pp7 + pp18 + pp27 + pp35 + pp10 + pp13;
    wire [5:0] columna4 = pp8 + pp19 + pp28 + pp36 + pp42 + pp47;
    wire [5:0] columna3 = pp9 + pp20 + pp29 + pp37 + pp43;
    wire [5:0] columna2 = pp10 + pp21 + pp13 + pp38 + pp44 + pp35;
    wire [5:0] columna1 = pp11 + pp22 + pp30 + pp39 + pp45 + pp9;
    assign P = (columna12 << 11) + (columna11 << 10) + (columna10 << 9) + (columna9 << 8) + (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule