// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//----------------------------------------------------------------------------
// Package: coralnpu_axi_slave_agent_pkg
// Description: Package for the CoralNPU AXI Slave Agent components.
//----------------------------------------------------------------------------
package coralnpu_axi_slave_agent_pkg;

  import uvm_pkg::*;
  `include "uvm_macros.svh"
  import transaction_item_pkg::*;
  import memory_map_pkg::*;

  //--------------------------------------------------------------------------
  // Class: coralnpu_axi_slave_model
  //--------------------------------------------------------------------------
  class coralnpu_axi_slave_model extends uvm_component;
    `uvm_component_utils(coralnpu_axi_slave_model)
    virtual coralnpu_axi_slave_if.TB_SLAVE_MODEL vif;

    function new(string name = "coralnpu_axi_slave_model",
                 uvm_component parent = null);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual coralnpu_axi_slave_if.TB_SLAVE_MODEL)::get(
          this, "", "vif", vif)) begin
        `uvm_fatal(get_type_name(),
                   "Virtual interface 'vif' not found for TB_SLAVE_MODEL")
      end
    endfunction

    virtual task run_phase(uvm_phase phase);
      vif.tb_slave_cb.awready <= 1'b0;
      vif.tb_slave_cb.wready  <= 1'b0;
      vif.tb_slave_cb.arready <= 1'b0;
      vif.tb_slave_cb.bvalid  <= 1'b0;
      vif.tb_slave_cb.rvalid  <= 1'b0;
      vif.tb_slave_cb.bresp   <= AXI_OKAY;
      vif.tb_slave_cb.rresp   <= AXI_OKAY;
      vif.tb_slave_cb.rdata   <= '0;
      fork
        handle_writes();
        handle_reads();
      join_none
    endtask

    // Slave agent: Handles AXI write transactions.
    //              - Internal Addrs (ITCM/DTCM): Error (Should stay internal) -> AXI_SLVERR
    //              - External Addrs: Error (No external memory exists) -> AXI_DECERR
    protected virtual task handle_writes();
      logic [IDWIDTH-1:0] current_bid;
      axi_resp_e resp;
      forever begin
        // Wait for write address
        vif.tb_slave_cb.awready <= 1'b0;
        @(vif.tb_slave_cb iff vif.tb_slave_cb.awvalid);
        current_bid = vif.tb_slave_cb.awid;
        `uvm_info(get_type_name(),
                  $sformatf("Slave Rcvd AW: Addr=0x%h ID=%0d",
                            vif.tb_slave_cb.awaddr, current_bid), UVM_HIGH)

        // Address Decoding / Filtering
        if (is_in_itcm(vif.tb_slave_cb.awaddr) || is_in_dtcm(vif.tb_slave_cb.awaddr)) begin
          `uvm_error(get_type_name(),
                     $sformatf("Internal Write Address leaked to External Bus: 0x%h",
                               vif.tb_slave_cb.awaddr))
          resp = AXI_SLVERR;
        end else begin
          `uvm_info(get_type_name(),
                    $sformatf("External Write Address (Unmapped): 0x%h",
                              vif.tb_slave_cb.awaddr), UVM_HIGH)
          resp = AXI_DECERR;
        end

        vif.tb_slave_cb.awready <= 1'b1;
        @(vif.tb_slave_cb);
        vif.tb_slave_cb.awready <= 1'b0;

        // Handle Write Data (Sink it)
        vif.tb_slave_cb.wready <= 1'b0;
        @(vif.tb_slave_cb iff vif.tb_slave_cb.wvalid);

        // Consume all write beats until WLAST
        vif.tb_slave_cb.wready <= 1'b1;
        // Wait for a valid last beat.
        do begin
          @(vif.tb_slave_cb);
        end while (!(vif.tb_slave_cb.wvalid && vif.tb_slave_cb.wlast));

        // Handshake happened at this cycle.
        vif.tb_slave_cb.wready <= 1'b0;

        // Send write response
        @(vif.tb_slave_cb);
        vif.tb_slave_cb.bvalid <= 1'b1;
        vif.tb_slave_cb.bresp  <= resp;
        vif.tb_slave_cb.bid    <= current_bid;

        do @(vif.tb_slave_cb); while (!vif.tb_slave_cb.bready);
        // Handshake happened at this cycle.
        vif.tb_slave_cb.bvalid <= 1'b0;
        `uvm_info(get_type_name(),
                  $sformatf("Slave Sent BResp %s ID=%0d",
                            resp.name(), current_bid), UVM_HIGH)
      end
    endtask

    // Slave agent: Handles AXI read transactions.
    //              - Internal Addrs (ITCM/DTCM): Error (Should stay internal) -> AXI_SLVERR
    //              - External Addrs: Error (No external memory exists) -> AXI_DECERR
    protected virtual task handle_reads();
      logic [IDWIDTH-1:0] current_rid;
      logic [7:0] current_len;
      axi_resp_e r_resp_val;

      forever begin
        // Wait for read address
        vif.tb_slave_cb.arready <= 1'b0;
        @(vif.tb_slave_cb iff vif.tb_slave_cb.arvalid);
        current_rid = vif.tb_slave_cb.arid;
        current_len = vif.tb_slave_cb.arlen;

        if (is_in_itcm(vif.tb_slave_cb.araddr) || is_in_dtcm(vif.tb_slave_cb.araddr)) begin
          `uvm_error(get_type_name(),
                     $sformatf("Internal Read Address leaked to External Bus: 0x%h",
                               vif.tb_slave_cb.araddr))
          r_resp_val = AXI_SLVERR;
        end else begin
          `uvm_info(get_type_name(),
                    $sformatf("External Read Address (Unmapped): 0x%h",
                              vif.tb_slave_cb.araddr), UVM_HIGH)
          r_resp_val = AXI_DECERR;
        end

        vif.tb_slave_cb.arready <= 1'b1;
        @(vif.tb_slave_cb);
        vif.tb_slave_cb.arready <= 1'b0;

        // Send Read Response (Burst)
        // Even for error responses, we must respect ARLEN and provide the
        // requested number of data transfers to avoid protocol violations.
        for (int i = 0; i <= current_len; i++) begin
            vif.tb_slave_cb.rvalid <= 1'b1;
            vif.tb_slave_cb.rresp  <= r_resp_val;
            vif.tb_slave_cb.rdata  <= 'x;
            vif.tb_slave_cb.rid    <= current_rid;

            if (i == current_len)
                vif.tb_slave_cb.rlast <= 1'b1;
            else
                vif.tb_slave_cb.rlast <= 1'b0;

            do @(vif.tb_slave_cb); while (!vif.tb_slave_cb.rready);
        end

        // Handshake for last beat finished.
        vif.tb_slave_cb.rvalid <= 1'b0;
        vif.tb_slave_cb.rlast  <= 1'b0;

        `uvm_info(get_type_name(),
                  $sformatf("Slave Sent RData %s ID=%0d Len=%0d",
                            r_resp_val.name(), current_rid, current_len),
                  UVM_HIGH)
      end
    endtask
  endclass

  //--------------------------------------------------------------------------
  // Class: coralnpu_axi_slave_agent
  //--------------------------------------------------------------------------
  class coralnpu_axi_slave_agent extends uvm_agent;
    `uvm_component_utils(coralnpu_axi_slave_agent)
    coralnpu_axi_slave_model slave_model;

    function new(string name = "coralnpu_axi_slave_agent",
                 uvm_component parent = null);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      slave_model = coralnpu_axi_slave_model::type_id::create("slave_model",
                                                             this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
    endfunction
  endclass

endpackage : coralnpu_axi_slave_agent_pkg
